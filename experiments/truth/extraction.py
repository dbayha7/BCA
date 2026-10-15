"""Room-to-win machinery of the D4RL-RT world: the Level-1 screen (R4a, no training) and actor extraction on the
frozen critic (R3b, R4b, R7; trains actors, so acceptance.py runs it only behind the pre-registration hash gate).

Arms differ only in the per-row behaviour-cloning multiplier v_i of TD3+BC's actor loss
    L(theta) = -lambda mean_i Q1(s_i, pi(s_i)) + mean_i v_i mean_a (pi(s_i) - a_i)^2,  lambda = alpha / mean|Q1|,
passed through algorithms.td3_bc.td3_bc_update's bc_multiplier argument, the hook BCA itself uses:
  NONE         v = 1 (also passed as an explicit vector, so every arm runs the same code path)
  P0(m)        v = m (even caution)
  OR-L-abs     v = 1 + beta M, M = the rank-normal multiset (s = 0.83, clip 2.5, mean 1;
               experiments/signal/d4rl_level1.assignments) placed by the rank of |e_A(s_i, a_i)|
  OR-L-signed  the same, placed by max(0, -e_A(s_i, a_i)): rows where the critic underrates the logged action
               relative to pi(s_i)
  SHUF_k       OR-L-abs's weights permuted (default_rng([seed, 10 + k])); ANTI its ranks reversed
with e_A(s, a) = e(s, a) - e(s, pi(s)), e = Q1 - Q^pi (truth from pool T's two rollouts per row). Ties in the
ranking variable are broken by one seeded permutation (default_rng([seed, 1])), never by row order.

Level-1 screen (Level1Screen, R4a). At the host actor theta_0 on pool T rows, with G(C) = sum_i (d pi(s_i) /
d theta)^T C_i (the actor's vjp; d4rl_level1's operator, float32 vjps summed in float64 per chunk):
    g_Q = G(-(lambda / N) grad_a Q1(s_i, pi_i)),  b0 = G((1 / N)(2 / d_a)(pi_i - a_i)),
    b_M = G((M_i / N)(2 / d_a)(pi_i - a_i)).
Level-1 strength rho(v) = ||g_BC[v]|| / ||g_Q||; rho(P0(m)) = m rho_host exactly. A weighted arm v = 1 + beta M
has g_BC = b0 + beta b_M, so its beta at rho* = m rho_host is the positive root of ||b0 + beta b_M|| = m ||b0||
(closed form, no ladder). Increment relative to NONE: Delta g = (m - 1) b0 for P0(m), beta b_M for a weighted arm.
The true policy gradient grad J = G_Pi(g_a(s) / N_Pi) over pool Pi states, g_a = grad_a Q^pi(s, a) at a = pi(s)
by central differences of rollout truth (the deterministic policy gradient, up to the 1 / (1 - gamma) visitation
factor). A gradient step theta - eta g changes J, relative to NONE's step, by -eta <grad J, Delta g> to first
order: the raw first-order true gain. The Adam-preconditioned gain is -<grad J, D Delta g>, D = 1 / (sqrt(v_hat)
+ 1e-8) from the host checkpoint's actor Adam second moment (bias-corrected); a host whose Adam count is 0 (a
random-init synthetic host) uses v_hat = (g_Q + b0)^2 and says so. The cosine cos(Delta g_OR, Delta g_P0) =
cos(b_M, b0) feeds R7(d).

Extraction (Extractor). The host's actor starts from its checkpoint params with a FRESH Adam (runtime.networks
.torch_adam at the host lr); the critic is frozen by a zeroed optimizer (optax.set_to_zero, as lq_harness
does), so td3_bc_update runs unchanged: its critic step adds zeros, its actor step climbs the frozen Q1 with the
arm's bc_multiplier. Every call uses an even iteration counter, so every call is an actor update (policy_freq
2): `steps` calls = `steps` actor updates. Batches of `batch` rows are drawn uniformly from pool T with
jax.random.randint, keys from PRNGKey(seed) split per update; arms of one unit (critic, seed) share every batch
index and smoothing-noise draw (common random numbers). jax.lax.scan blocks of `block` updates; the critic
params are checked unchanged after every run.

Strength S (strength) = sqrt(mean_s |pi_arm(s) - pi_NONE(s)|^2) over the host's withheld states (up to
`s_states` seeded): the design's RMS distance from NONE. Matching (match_ladder): a weighted arm is trained at a
5-point beta ladder, beta* interpolates log S against log beta between the bracketing points (linearly from
(0, 0) below the first point), and one rerun at beta* gives the matched policy; fidelity |S / S* - 1| is
recorded. An unbracketed target is unreachable (never extrapolated).

Evaluation (evaluate): the D4RL-normalized return of each policy over one reset bank shared by all arms, as
pools.policy_episodes episodes (bare MjSims, state_restore.Physics.step = env.step bit for bit, time limit
1,000). This replaces runtime/common.eval_policy's vector environment: same dynamics and policy, faster, and the
same start states for every arm.
"""

import copy
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.truth.state_restore import require  # noqa: E402

KEY_EXTRACT = 1163416660
VJP_BLOCK = 8  # cotangent sets per compiled vjp call


def oracle_weights(e_a, kind, seed, *, s=0.83, clip=2.5, n_shuf=0):
    """(SIG, SHUF list, ANTI) rank-normal weights for an oracle: kind 'abs' ranks |e_A|, 'signed' max(0, -e_A)."""
    from experiments.signal.d4rl_level1 import assignments, shuffle_permutations

    e_a = np.asarray(e_a, np.float64)
    require(np.isfinite(e_a).all(), "oracle needs finite e_A")
    x = np.abs(e_a) if kind == "abs" else np.maximum(0.0, -e_a)
    require(kind in ("abs", "signed"), "oracle kind is abs or signed")
    tie = np.random.default_rng([seed, 1]).permutation(len(x))
    shuf = shuffle_permutations(len(x), seed, n_shuf)
    assigned, m = assignments(x, tie, shuf, [], s, clip)
    return assigned["SIG"], assigned["SHUF"], assigned["ANTI"], m


def with_params(policy, params):
    """A copy of a FrozenPolicy that reuses its compiled program with other actor params."""
    p = copy.copy(policy)
    p.params = params
    p._identity = None
    return p


def _chunks(n, size):
    return [(a, min(n, a + size)) for a in range(0, n, size)]


def _pad(x, size):
    out = np.zeros((size,) + x.shape[1:], x.dtype)
    out[:len(x)] = x
    return out


# ---------------------------------------------------------------- Level-1 screen (R4a)

class Level1Screen:
    """Exact first-order quantities at the host actor on pool T rows, scored against pool Pi's true gradient."""

    def __init__(self, host, obs_T, act_T, obs_Pi, grad_a_Pi, chunk=4096):
        import jax
        import jax.numpy as jnp
        from jax.flatten_util import ravel_pytree

        actor, critic = host.actor, host.critic
        params, cparams = host.native.actor.params, host.native.critic.params
        self.host, self.chunk = host, int(chunk)
        self.d = host.task.action_dim

        def rows(s):
            pi = actor.apply(params, s)
            q1 = critic.apply(cparams, s, pi)[..., 0]
            qa = jax.grad(lambda a: critic.apply(cparams, s, a)[..., 0].sum())(pi)
            return pi, q1, qa

        def vjp(s, cot):
            _, pull = jax.vjp(lambda p: actor.apply(p, s), params)
            return jax.vmap(lambda c: ravel_pytree(pull(c)[0])[0])(cot)

        self._rows, self._vjp = jax.jit(rows), jax.jit(vjp)
        self.sT = host.standardized(obs_T)
        self.aT = np.asarray(act_T, np.float32)
        pis, q1s, qas = [], [], []
        for a, b in _chunks(len(self.sT), self.chunk):
            pi, q1, qa = self._rows(_pad(self.sT[a:b], self.chunk))
            pis.append(np.asarray(pi)[:b - a])
            q1s.append(np.asarray(q1)[:b - a])
            qas.append(np.asarray(qa)[:b - a])
        self.pi, self.q1, self.qa = np.concatenate(pis), np.concatenate(q1s), np.concatenate(qas)
        self.N = len(self.sT)
        self.alpha = float(host.args.alpha)
        self.lam = self.alpha / float(np.mean(np.abs(self.q1.astype(np.float64))))
        flat, _ = ravel_pytree(params)
        self.P = int(flat.size)
        self.bc_dir = ((2.0 / self.d) * (self.pi.astype(np.float64) - self.aT.astype(np.float64))) / self.N
        g = self.G(np.stack([-(self.lam / self.N) * self.qa.astype(np.float64), self.bc_dir]))
        self.g_Q, self.b0 = g[0], g[1]
        self.rho_host = float(np.linalg.norm(self.b0) / np.linalg.norm(self.g_Q))
        sPi = host.standardized(obs_Pi)
        ga = np.asarray(grad_a_Pi, np.float64)
        require(ga.shape == (len(sPi), self.d) and np.isfinite(ga).all(), "pool Pi gradients must be finite (n, d)")
        self.grad_J = self.G(ga[None] / len(sPi), obs=sPi)[0]
        self.D, self.preconditioner = self._adam_diagonal(ravel_pytree)

    def G(self, cot, obs=None):
        """(K, P) float64: sum_i (d pi(s_i) / d theta)^T cot[k, i] over the rows (pool T unless obs is given)."""
        s = self.sT if obs is None else obs
        cot = np.asarray(cot, np.float64)
        K = len(cot)
        out = np.zeros((K, self.P))
        for k0 in range(0, K, VJP_BLOCK):
            block = np.zeros((VJP_BLOCK, len(s), self.d), np.float32)
            block[:min(VJP_BLOCK, K - k0)] = cot[k0:k0 + VJP_BLOCK]
            for a, b in _chunks(len(s), self.chunk):
                c = np.zeros((VJP_BLOCK, self.chunk, self.d), np.float32)
                c[:, :b - a] = block[:, a:b]
                part = np.asarray(self._vjp(_pad(s[a:b], self.chunk), c), np.float64)
                out[k0:k0 + VJP_BLOCK] += part[:min(VJP_BLOCK, K - k0)]
        return out

    def _adam_diagonal(self, ravel_pytree):
        state = self.host.native.actor.opt_state
        adam = state[0]
        count = int(np.asarray(adam.count))
        if count > 0:
            nu = np.asarray(ravel_pytree(adam.nu)[0], np.float64)
            v_hat = nu / (1.0 - 0.999 ** count)
            how = f"host checkpoint actor Adam second moment, count {count}, bias-corrected"
        else:
            v_hat = (self.g_Q + self.b0) ** 2
            how = "Adam count 0 (random-init host): v_hat = (g_Q + b0)^2, the NONE gradient squared"
        return 1.0 / (np.sqrt(v_hat) + 1e-8), how

    def b_of(self, weights):
        """b_M = G((M_i / N)(2 / d_a)(pi_i - a_i)) for each weight vector, (K, P)."""
        w = np.asarray(weights, np.float64)
        return self.G(w[:, :, None] * self.bc_dir[None])

    def match_beta(self, b_m, level):
        """beta > 0 with ||b0 + beta b_M|| = level ||b0|| (level > 1)."""
        require(level > 1, "matching level must exceed 1")
        aa = float(b_m @ b_m)
        ab = float(self.b0 @ b_m)
        cc = float(self.b0 @ self.b0) * (1.0 - level ** 2)
        if aa <= 0:
            return None
        disc = ab * ab - aa * cc
        beta = (-ab + np.sqrt(disc)) / aa
        return float(beta) if beta > 0 else None

    def gains(self, delta_g):
        return dict(raw=float(-(self.grad_J @ delta_g)), adam=float(-(self.grad_J @ (self.D * delta_g))))

    @staticmethod
    def cosine(x, y):
        nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
        return float(x @ y / (nx * ny)) if nx > 0 and ny > 0 else None


# ---------------------------------------------------------------- extraction (trains actors)

class Extractor:
    """Actor-only TD3+BC on the frozen critic over pool T, through td3_bc_update with bc_multiplier."""

    def __init__(self, host, pool_converted_ids, *, batch, block, lr=None):
        import jax
        import jax.numpy as jnp
        import optax
        from flax.training.train_state import TrainState

        from algorithms.td3_bc import td3_bc_update
        from runtime.networks import Transition, torch_adam

        self.host = host
        ids = np.asarray(pool_converted_ids, np.int64)
        conv = host.converted
        self.data = Transition(jnp.asarray(host.standardized(conv["observations"][ids])),
                               jnp.asarray(conv["actions"][ids], jnp.float32),
                               jnp.asarray(conv["rewards"][ids], jnp.float32),
                               jnp.asarray(host.standardized(conv["next_observations"][ids])),
                               jnp.asarray(conv["terminals"][ids], jnp.float32))
        self.n = len(ids)
        self.args = replace(host.args, batch_size=int(batch), lr=float(host.args.lr if lr is None else lr))
        self.block = int(block)
        args, actor, critic, data, n = self.args, host.actor, host.critic, self.data, self.n
        m = host.max_action
        native = host.native
        zero, adam = optax.set_to_zero(), torch_adam(args.lr)  # created once: they are static parts of the jit key
        self._initial = lambda: native._replace(
            actor=TrainState.create(apply_fn=actor.apply, params=native.actor.params, tx=adam),
            critic=native.critic.replace(tx=zero, opt_state=zero.init(native.critic.params)))

        def run(state, key, it, v):
            def body(carry, _):
                state, key, it = carry
                key, kb, kn = jax.random.split(key, 3)
                idx = jax.random.randint(kb, (args.batch_size,), 0, n)
                batch = jax.tree_util.tree_map(lambda x: x[idx], data)
                state, metrics = td3_bc_update(args, actor.apply, critic.apply, state, batch, it, kn, m,
                                               bc_multiplier=v[idx])
                return (state, key, it + args.policy_freq), (metrics["actor_loss"], metrics["q_mean"],
                                                             metrics["bc_loss"])
            return jax.lax.scan(body, (state, key, it), None, length=self.block)

        self._run = jax.jit(run)
        self._jax, self._jnp = jax, jnp

    def train(self, v, seed, steps):
        """Actor params after `steps` actor updates with per-row multiplier v (pool T order)."""
        jax, jnp = self._jax, self._jnp
        v = np.asarray(v, np.float32)
        require(v.shape == (self.n,) and np.isfinite(v).all() and np.all(v >= 0), "one finite v >= 0 per pool row")
        require(type(steps) is int and steps > 0 and steps % self.block == 0, "steps must be a multiple of block")
        state = self._initial()
        key = jax.random.fold_in(jax.random.PRNGKey(int(seed)), KEY_EXTRACT)
        it = jnp.int32(0)
        losses = []
        for _ in range(steps // self.block):
            (state, key, it), (actor_loss, q_mean, bc) = self._run(state, key, it, jnp.asarray(v))
            losses.append((float(np.mean(actor_loss)), float(np.mean(q_mean)), float(np.mean(bc))))
        frozen = all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(
            jax.tree_util.tree_leaves(state.critic.params), jax.tree_util.tree_leaves(self.host.native.critic.params)))
        require(frozen, "the critic moved during extraction")
        last = losses[-1]
        return state.actor.params, dict(actor_updates=int(steps), final_block_actor_loss=last[0],
                                        final_block_q_mean=last[1], final_block_bc_loss=last[2], critic_frozen=True)


class PolicyStats:
    """pi(s) and Q1(s, pi(s)) on a fixed set of standardized states, for S and R3b's Q1 rise."""

    def __init__(self, host, obs, chunk=4096):
        import jax

        self.host, self.chunk = host, int(chunk)
        self.s = host.standardized(obs)
        actor, critic, cparams = host.actor, host.critic, host.native.critic.params
        self._fn = jax.jit(lambda p, s: (actor.apply(p, s), critic.apply(cparams, s, actor.apply(p, s))[..., 0]))

    def __call__(self, params):
        pis, qs = [], []
        for a, b in _chunks(len(self.s), self.chunk):
            pi, q = self._fn(params, _pad(self.s[a:b], self.chunk))
            pis.append(np.asarray(pi)[:b - a])
            qs.append(np.asarray(q)[:b - a])
        return np.concatenate(pis).astype(np.float64), np.concatenate(qs).astype(np.float64)


def strength(pi_arm, pi_none):
    """S = sqrt(mean_s |pi_arm(s) - pi_NONE(s)|^2)."""
    return float(np.sqrt(np.mean(np.sum((np.asarray(pi_arm) - np.asarray(pi_none)) ** 2, axis=1))))


def match_ladder(ladder, s_values, target):
    """beta* with S(beta*) = target from a ladder (ascending betas): log-log interpolation between bracketing
    points, linear from (0, 0) below the first; None when the target is not bracketed."""
    ladder, s_values = np.asarray(ladder, np.float64), np.asarray(s_values, np.float64)
    require(np.all(np.diff(ladder) > 0) and np.all(ladder > 0), "ascending positive ladder")
    above = np.flatnonzero(s_values >= target)
    if not len(above) or target <= 0:
        return None, dict(bracketed=False, monotone=bool(np.all(np.diff(s_values) >= 0)))
    j = int(above[0])
    if j == 0:
        beta = ladder[0] * target / s_values[0] if s_values[0] > 0 else None
        how = "linear from (0, 0)"
    else:
        lo, hi = s_values[j - 1], s_values[j]
        if lo <= 0:
            beta = ladder[j - 1] + (ladder[j] - ladder[j - 1]) * (target - lo) / (hi - lo)
            how = "linear (lower S is 0)"
        else:
            w = (np.log(target) - np.log(lo)) / (np.log(hi) - np.log(lo)) if hi > lo else 1.0
            beta = float(np.exp(np.log(ladder[j - 1]) + w * (np.log(ladder[j]) - np.log(ladder[j - 1]))))
            how = "log-log between bracketing points"
    return (None if beta is None else float(beta)), dict(bracketed=True, interval=int(j), rule=how,
                                                          monotone=bool(np.all(np.diff(s_values) >= 0)))


def evaluate(physics, policy, params, qpos, qvel, dataset, *, horizon=1000, every=None, sims=None):
    """Normalized returns of `params` on the reset bank (qpos, qvel); optionally the visited states."""
    from experiments.truth.pools import normalized_scores, policy_episodes

    ep, rec = policy_episodes(physics, with_params(policy, params), qpos, qvel, horizon=horizon, every=every,
                              sims=sims)
    return normalized_scores(dataset, ep["returns"]), ep, rec

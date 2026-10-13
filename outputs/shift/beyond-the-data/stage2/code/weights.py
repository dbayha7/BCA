"""Stage 2, the weighted version only (the declared fallback): the likelihood ratio w(s, a) between the actor's
(state, command) pairs and the recorded pairs, from a classifier.

The estimator is fixed here, before any BCA run, and restates the classifier of the classifier-ratio stand-in
(scratchpad/ood/standins/standins.py, plan section 4.2: "the score-free part of Variant A-weighted"), so that the
weighted version and its data-only rival read the same ratio:

  - network: Dense 64 - ReLU - Dense 64 - ReLU - Dense 1 (the shape of BCA's scale network), output the logit, so
    log w = logit = log(g / (1 - g)); inputs the normalised observation / (1 + 0.001) and the command;
  - target pairs (s_i, clip(pi(s_i) + a_i - c(s_i), -1, 1)) labelled 1 against recorded pairs (s_i, a_i) labelled 0,
    at the same training rows (the recorder's own scatter re-centred on the actor's command; c the cloned actor);
  - balanced logistic loss, 2,000 Adam steps (runtime.networks.torch_adam, learning rate 0.001) of 512 rows from the
    fixed stream default_rng([20261015, seed, 82]), from the fixed initialisation of default_rng([20261015, seed,
    84]);
  - refitted with the actor whose pairs are the target: the gapped critic's actor in the detection test, each arm's
    own actor before every 4,000-step chunk in the policy test, the current actor at every BCA refresh in the
    co-trained host.

If the locked detection script's classifier-ratio estimator differs from this one, the pre-registration must be
rewritten (before approval) so that both use one estimator (detect_contract.py says what that script must expose).

Pure functions of arrays and parameters: nothing here loads ThrusterChain data.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/user/bca")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import flax.linen as nn  # noqa: E402

import runtime.networks as RN  # noqa: E402

CLS_STEPS, CLS_BATCH, CLS_LR, CLS_HIDDEN = 2000, 512, 1e-3, 64
TAG_CLS_ROWS, TAG_CLS_INIT = 82, 84
LEARNER_SEED = 20261015


class ClassifierNet(nn.Module):
    """g's logit: Dense 64 - ReLU - Dense 64 - ReLU - Dense 1 on (obs / 1.001, command)."""

    hidden: int = CLS_HIDDEN

    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate([obs / (1.0 + 0.001), action], axis=-1)
        x = nn.relu(nn.Dense(self.hidden)(x))
        x = nn.relu(nn.Dense(self.hidden)(x))
        return nn.Dense(1)(x).squeeze(-1)


CLS = ClassifierNet()


def rng_np(rep, *tags):
    return np.random.default_rng([LEARNER_SEED, int(rep), *[int(t) for t in tags]])


def cls_init(rep, dx, da):
    seed = int(rng_np(rep, TAG_CLS_INIT).integers(2 ** 31 - 1))
    return CLS.init(jax.random.PRNGKey(seed), jnp.zeros(dx), jnp.zeros(da)), seed


def cls_rows(rep, n, steps=CLS_STEPS, batch=CLS_BATCH):
    return rng_np(rep, TAG_CLS_ROWS).integers(n, size=(steps, batch))


def logit(params, obs, action):
    return CLS.apply(params, obs, action)


def target_commands(pi, recorded, cloned):
    """The target pairs' commands: the recorder's scatter around the cloned actor, re-centred on the actor."""
    return jnp.clip(pi + recorded - cloned, -1.0, 1.0)


def fit(p0, obs, recorded, target, rows, lr=CLS_LR):
    """Balanced logistic fit: target pairs labelled 1, recorded pairs 0, at the same rows; Adam (torch_adam).
    Returns (params, per-step losses). Traceable (jit/vmap over the actor's target commands)."""
    tx = RN.torch_adam(lr)

    def loss(p, r):
        lt = CLS.apply(p, obs[r], target[r])
        lr_ = CLS.apply(p, obs[r], recorded[r])
        return jnp.mean(jax.nn.softplus(-lt)) + jnp.mean(jax.nn.softplus(lr_))

    def body(carry, r):
        p, s = carry
        val, g = jax.value_and_grad(loss)(p, r)
        u, s = tx.update(g, s, p)
        return (jax.tree_util.tree_map(lambda x, y: x + y, p, u), s), val

    (p, _), ls = jax.lax.scan(body, (p0, tx.init(p0)), rows)
    return p, ls


_FIT = {}


def fit_for_actor(actor_apply, actor_params, obs, recorded, cloned, rows, p0):
    """One classifier fit with one actor (jit-compiled once per shape)."""
    key = (id(actor_apply), tuple(np.shape(obs)), tuple(np.shape(rows)))
    if key not in _FIT:
        def f(ap, o, rec, cl, r, p):
            pi = actor_apply(ap, o)
            return fit(p, o, rec, target_commands(pi, rec, cl), r)
        _FIT[key] = jax.jit(f)
    p, ls = _FIT[key](actor_params, jnp.asarray(obs), jnp.asarray(recorded), jnp.asarray(cloned),
                      jnp.asarray(rows, jnp.int32), p0)
    ls = np.asarray(ls, np.float64)
    return p, dict(loss_first100=float(ls[:100].mean()), loss_last100=float(ls[-100:].mean()),
                   finite=bool(np.all(np.isfinite(ls))))

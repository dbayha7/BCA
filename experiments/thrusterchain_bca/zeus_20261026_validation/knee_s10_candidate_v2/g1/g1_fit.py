"""The supervised fit of G1 (G1_RULE.md): Unifloral's SoftQNetwork, line for line, fit by mean-squared error to
exact labels with Adam(3e-4, eps 1e-5), batch 256 drawn uniformly with replacement, on the CPU JAX runtime.

`upstream_match()` proves the critic class below equals the pinned upstream text (td3_bc.py, sha256 6f5ab6f9...).
"""
import inspect
import os
from pathlib import Path
import re

os.environ.setdefault("JAX_PLATFORMS", "cpu")                 # the rule: the CPU runtime, whatever else is installed

import jax                                                   # noqa: E402
import jax.numpy as jnp                                      # noqa: E402
import flax.linen as nn                                      # noqa: E402
import numpy as np                                           # noqa: E402
import optax                                                 # noqa: E402

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE.parents[2] / "titan_20261026/payload/unifloral_td3bc_validation_v1/upstream/algorithms/td3_bc.py"
UPSTREAM_SHA256 = "6f5ab6f98d706d902a18f11e1c81e99e2691dc4c8a53843bf6521c9f95d00eee"
LR, EPS, BATCH = 3e-4, 1e-5, 256
CHUNK = 1000


class SoftQNetwork(nn.Module):
    obs_mean: jax.Array
    obs_std: jax.Array

    @nn.compact
    def __call__(self, obs, action):
        obs = (obs - self.obs_mean) / (self.obs_std + 1e-3)
        x = jnp.concatenate([obs, action], axis=-1)
        for _ in range(2):
            x = nn.Dense(256)(x)
            x = nn.relu(x)
        q = nn.Dense(1)(x)
        return q.squeeze(-1)


def _normalized(text):
    return [line.rstrip() for line in text.splitlines() if line.strip()]


def upstream_match():
    """True when the SoftQNetwork source above equals the upstream class body, line for line (blank lines ignored)."""
    import hashlib
    raw = UPSTREAM.read_bytes()
    if hashlib.sha256(raw).hexdigest() != UPSTREAM_SHA256:
        return False
    text = raw.decode("utf-8")
    match = re.search(r"class SoftQNetwork\(nn\.Module\):.*?(?=\nclass )", text, re.S)
    return match is not None and _normalized(match.group(0)) == _normalized(inspect.getsource(SoftQNetwork))


def normalizer(obs):
    """Mean and SD of float32 observations, as float32 arrays (the host's obs_mean, obs_std)."""
    obs = np.asarray(obs, np.float32)
    return obs.mean(axis=0, dtype=np.float64).astype(np.float32), obs.std(axis=0, dtype=np.float64).astype(np.float32)


def fit(obs, actions, labels, obs_mean, obs_std, seed, checkpoints, *, batch=BATCH, lr=LR, eps=EPS, chunk=CHUNK):
    """Returns ({updates: host params}, {updates: mean loss of the last chunk ending there}) for the checkpoints."""
    obs_d = jnp.asarray(np.asarray(obs, np.float32))
    act_d = jnp.asarray(np.asarray(actions, np.float32))
    lab_d = jnp.asarray(np.asarray(labels, np.float32))
    n = int(obs_d.shape[0])
    checkpoints = sorted(int(c) for c in checkpoints)
    if not checkpoints or any(c <= 0 or c % chunk for c in checkpoints):
        raise ValueError("Checkpoints must be positive multiples of the chunk")
    net = SoftQNetwork(jnp.asarray(np.asarray(obs_mean, np.float32)), jnp.asarray(np.asarray(obs_std, np.float32)))
    key, k_init = jax.random.split(jax.random.PRNGKey(int(seed)))
    params = net.init(k_init, obs_d[:1], act_d[:1])
    tx = optax.adam(lr, eps=eps)
    opt = tx.init(params)

    def step(carry, _, data):
        params, opt, key = carry
        o, a, y = data
        key, sub = jax.random.split(key)
        idx = jax.random.randint(sub, (batch,), 0, n)

        def loss_fn(p):
            return jnp.mean((net.apply(p, o[idx], a[idx]) - y[idx]) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt = tx.update(grads, opt, params)
        return (optax.apply_updates(params, updates), opt, key), loss

    @jax.jit
    def run_chunk(params, opt, key, o, a, y):
        (params, opt, key), losses = jax.lax.scan(lambda c, x: step(c, x, (o, a, y)), (params, opt, key), None, length=chunk)
        return params, opt, key, losses

    out, curve, done = {}, {}, 0
    while done < checkpoints[-1]:
        params, opt, key, losses = run_chunk(params, opt, key, obs_d, act_d, lab_d)
        done += chunk
        curve[done] = float(jnp.mean(losses))
        if done in checkpoints:
            out[done] = jax.device_get(params)
    return out, curve


def q_values(params, obs_mean, obs_std, obs, actions):
    """Q1 of one fitted head at (obs, actions) rows, float64 on the host."""
    net = SoftQNetwork(jnp.asarray(np.asarray(obs_mean, np.float32)), jnp.asarray(np.asarray(obs_std, np.float32)))
    q = net.apply(params, jnp.asarray(np.asarray(obs, np.float32)), jnp.asarray(np.asarray(actions, np.float32)))
    return np.asarray(jax.device_get(q), np.float64)


def params_bytes(params):
    from flax import serialization
    return serialization.to_bytes(params)

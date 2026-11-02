"""Independent fixed-fixture TD3+BC contracts; see the predeclared plan."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from algorithms.td3_bc import td3_bc_update
from runtime.networks import Transition


class State(NamedTuple):
    actor: TrainState
    actor_target: TrainState
    critic: TrainState
    critic_target: TrainState


ATOL, RTOL = 1e-6, 1e-5
ADAM_EPS = 1e-8
args = SimpleNamespace(policy_noise=0.17, noise_clip=0.22, discount=0.9,
                       alpha=1.7, tau=0.2, policy_freq=2)
obs = np.array([[0.2, -0.6], [0.7, 0.3], [-0.4, 0.8]], np.float32)
nxt = np.array([[0.25, -0.3], [-0.2, 0.5], [0.6, -0.7]], np.float32)
action = np.array([[0.1, -0.3], [-0.4, 0.2], [0.3, 0.6]], np.float32)
reward = np.array([0.4, -0.6, 0.2], np.float32)
done = np.array([0, 1, 0], np.float32)
actor = {"w": np.array([[0.4, -0.2], [0.3, 0.5]], np.float32),
         "b": np.array([0.12, -0.08], np.float32)}
actor_tgt = {"w": np.array([[0.35, -0.15], [0.25, 0.45]], np.float32),
             "b": np.array([0.1, -0.04], np.float32)}
critic = {"ws": np.array([[0.4, -0.3], [0.2, 0.5]], np.float32),
          "wa": np.array([[0.6, -0.4], [-0.3, 0.2]], np.float32),
          "b": np.array([0.7, -0.5], np.float32)}
critic_tgt = {"ws": np.array([[0.35, -0.25], [0.15, 0.45]], np.float32),
              "wa": np.array([[0.55, -0.35], [-0.25, 0.15]], np.float32),
              "b": np.array([0.65, -0.45], np.float32)}
actor_direction = {"w": np.array([[0.3, -0.2], [0.1, 0.4]], np.float64),
                   "b": np.array([-0.2, 0.2], np.float64)}
critic_direction = {"ws": np.array([[0.2, -0.3], [0.1, 0.4]], np.float64),
                    "wa": np.array([[-0.1, 0.2], [0.3, -0.2]], np.float64),
                    "b": np.array([0.15, -0.1], np.float64)}


def actor_apply(p, x):
    return jnp.tanh(x @ p["w"] + p["b"])


def critic_apply(p, x, a):
    return x @ p["ws"] + a @ p["wa"] + p["b"]


def as64(tree):
    return {k: np.asarray(v, dtype=np.float64) for k, v in tree.items()}


def add_scaled(p, d, amount):
    return {k: p[k] + amount * d[k] for k in p}


def dot_tree(a, b):
    return sum(float(np.sum(a[k] * b[k])) for k in a)


def np_policy(p, x):
    return np.tanh(x @ p["w"] + p["b"])


def np_q(p, x, a):
    return x @ p["ws"] + a @ p["wa"] + p["b"]


def adam_first(p, g, lr):
    return {k: p[k] - lr * g[k] / (np.abs(g[k]) + ADAM_EPS) for k in p}


def reference(noise, update_actor):
    x, xn, a, r, d = map(lambda z: np.asarray(z, dtype=np.float64), (obs, nxt, action, reward, done))
    pa, pt, pc, pct = map(as64, (actor, actor_tgt, critic, critic_tgt))
    unclip_noise = np.asarray(noise, np.float64) * args.policy_noise
    clipped_noise = np.clip(unclip_noise, -args.noise_clip, args.noise_clip)
    unclip_action = np_policy(pt, xn) + clipped_noise
    next_action = np.clip(unclip_action, -1.0, 1.0)
    target_heads = np_q(pct, xn, next_action)
    target = r + (1.0 - d) * args.discount * np.min(target_heads, axis=1)
    q = np_q(pc, x, a)
    errors = q - target[:, None]
    critic_loss = np.sum(np.mean(errors * errors, axis=0))
    critic_grad = {"ws": x.T @ (2 * errors / len(x)),
                   "wa": a.T @ (2 * errors / len(x)),
                   "b": np.sum(2 * errors / len(x), axis=0)}
    pc_new = adam_first(pc, critic_grad, 0.002)
    pi = np_policy(pa, x)
    q1 = np_q(pc_new, x, pi)[:, 0]
    lam = args.alpha / np.mean(np.abs(q1))
    bc = np.mean((pi - a) ** 2)
    actor_loss = -lam * np.mean(q1) + bc
    dpi = -lam * pc_new["wa"][:, 0][None, :] / len(x) + 2 * (pi - a) / pi.size
    dz = dpi * (1 - pi * pi)
    actor_grad = {"w": x.T @ dz, "b": np.sum(dz, axis=0)}
    pa_new = adam_first(pa, actor_grad, 0.001) if update_actor else pa
    actor_t_new = {k: args.tau * pa_new[k] + (1 - args.tau) * pt[k] for k in pt} if update_actor else pt
    critic_t_new = {k: args.tau * pc_new[k] + (1 - args.tau) * pct[k] for k in pct} if update_actor else pct

    # Exclude nondifferentiable target-critic ties, action/noise clipping edges,
    # actor normalizer singularities and near-zero first-step Adam gradients.
    assert np.min(np.abs(q1)) > 0.05
    assert np.min(np.abs(target_heads[:, 0] - target_heads[:, 1])) > 0.05
    assert np.min(np.abs(np.abs(unclip_action) - 1)) > 0.05
    assert np.min(np.abs(np.abs(unclip_noise) - args.noise_clip)) > 0.001
    assert min(np.min(np.abs(v)) for v in actor_grad.values()) > 1e-5
    assert min(np.min(np.abs(v)) for v in critic_grad.values()) > 1e-5

    def actor_obj(p):
        candidate = np_policy(p, x)
        candidate_q = np_q(pc_new, x, candidate)[:, 0]
        return -lam * np.mean(candidate_q) + np.mean((candidate - a) ** 2)

    def critic_obj(p):
        e = np_q(p, x, a) - target[:, None]
        return np.sum(np.mean(e * e, axis=0))

    derivatives = {}
    for label, p, direction, grad, obj in (("actor", pa, actor_direction, actor_grad, actor_obj),
                                           ("critic", pc, critic_direction, critic_grad, critic_obj)):
        exact = dot_tree(grad, direction)
        fds = [(obj(add_scaled(p, direction, h)) - obj(add_scaled(p, direction, -h))) / (2*h)
               for h in (0.001, 0.002)]
        for approx in fds:
            np.testing.assert_allclose(approx, exact, atol=ATOL, rtol=RTOL)
        derivatives[label] = {"analytical": exact, "central_differences": fds}
    return {"actor": pa_new, "actor_target": actor_t_new, "critic": pc_new,
            "critic_target": critic_t_new, "actor_grad": actor_grad, "critic_grad": critic_grad,
            "target": target, "target_heads": target_heads, "next_action": next_action,
            "metrics": {"critic_loss": critic_loss, "actor_loss": actor_loss if update_actor else 0.,
                        "q_mean": np.mean(q1) if update_actor else 0., "lambda": lam if update_actor else 0.,
                        "bc_loss": bc if update_actor else 0.}, "directional_derivatives": derivatives}


def make_state():
    def create(params, lr, apply_fn):
        return TrainState.create(apply_fn=apply_fn, params=jax.tree_util.tree_map(jnp.asarray, params),
                                 tx=optax.adam(lr, b1=0.9, b2=0.999, eps=ADAM_EPS, eps_root=0.))
    return State(create(actor, 0.001, actor_apply), create(actor_tgt, 0.001, actor_apply),
                 create(critic, 0.002, critic_apply), create(critic_tgt, 0.002, critic_apply))


def check():
    transition = Transition(*[jnp.asarray(x) for x in (obs, action, reward, nxt, done)])
    noise_key = jax.random.PRNGKey(20261027)
    noise = np.asarray(jax.random.normal(noise_key, action.shape))
    source = Path(__file__)
    plan = source.with_name("COMPILED_IMPLEMENTATION_PLAN_20261027.md")
    result = {"plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "jax_version": jax.__version__, "backend": jax.default_backend(),
              "device": str(jax.devices()[0]), "atol": ATOL, "rtol": RTOL,
              "rng_key": np.asarray(noise_key).tolist(), "mask": done.tolist(), "cases": {}}
    for it in (1, 2):
        ref = reference(noise, it == 2)
        case = {"target": ref["target"].tolist(), "target_heads": ref["target_heads"].tolist(),
                "next_action": ref["next_action"].tolist(), "directional_derivatives": ref["directional_derivatives"],
                "execution": {}}
        for mode in ("native", "jit"):
            state = make_state()
            update = lambda s: td3_bc_update(args, actor_apply, critic_apply, s, transition, it, noise_key, 1.)
            result_state, metrics = (jax.jit(update) if mode == "jit" else update)(state)
            max_abs = {}
            for key in ("actor", "actor_target", "critic", "critic_target"):
                observed, expected = getattr(result_state, key), ref[key]
                expected_step = 1 if (key == "critic" or it == 2) else 0
                assert int(observed.step) == expected_step, (mode, it, key, observed.step)
                for leaf in expected:
                    actual = np.asarray(observed.params[leaf])
                    np.testing.assert_allclose(actual, expected[leaf], atol=ATOL, rtol=RTOL,
                                               err_msg=f"{mode} it={it} {key}.{leaf}")
                    max_abs[f"{key}.{leaf}"] = float(np.max(np.abs(actual - expected[leaf])))
            for key, expected in ref["metrics"].items():
                np.testing.assert_allclose(np.asarray(metrics[key]), expected, atol=ATOL, rtol=RTOL,
                                           err_msg=f"{mode} it={it} metric {key}")
            for name in ("critic", "actor"):
                if name == "actor" and it == 1:
                    continue
                # From zero moments, Adam mu = (1-beta1) * gradient.
                mu = getattr(result_state, name).opt_state[0].mu
                for leaf, expected in ref[f"{name}_grad"].items():
                    observed = np.asarray(mu[leaf]) / 0.1
                    np.testing.assert_allclose(observed, expected, atol=ATOL, rtol=RTOL,
                                               err_msg=f"{mode} it={it} {name} gradient {leaf}")
                    max_abs[f"{name}_gradient.{leaf}"] = float(np.max(np.abs(observed - expected)))
            case["execution"][mode] = {"max_abs_errors": max_abs,
                                        "metrics": {k: float(np.asarray(v)) for k, v in metrics.items()}}
        result["cases"][str(it)] = case
    return result


if __name__ == "__main__":
    print(json.dumps(check(), indent=2, allow_nan=False))

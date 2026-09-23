"""Common."""

import collections
import collections.abc

if not hasattr(collections, "Mapping"):
    collections.Mapping = collections.abc.Mapping
    collections.MutableMapping = collections.abc.MutableMapping
import math
import os
import sys
import warnings
from collections import namedtuple
from dataclasses import asdict
from datetime import datetime

os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
os.environ.setdefault("MUJOCO_GL", "egl")
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
import runtime.kitchen as _kitchen_compat
import d4rl

try:
    import d4rl.gym_mujoco
except Exception:
    pass
try:
    import d4rl.locomotion
except Exception:
    pass
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as onp
import optax

_SILENCED_THIRD_PARTY = (
    ".*Box bound precision lowered by casting to float32.*",
    ".*Function `env\\.seed\\(seed\\)` is marked as deprecated.*",
    ".*Function `rng\\.randn\\(\\*size\\)` is marked as deprecated.*",
)
for _msg in _SILENCED_THIRD_PARTY:
    warnings.filterwarnings("ignore", message=_msg)
from runtime.rewards import (
    DEFAULT_BIAS as RT_DEFAULT_BIAS,
    DEFAULT_MODE as RT_DEFAULT_MODE,
    DEFAULT_SCALE as RT_DEFAULT_SCALE,
    apply_reward_transform,
    check_reward_transform,
    is_antmaze,
)
from runtime.networks import Transition, TransitionNA


def compute_mean_std(states, eps=0.001):
    mean = states.mean(0)
    std = states.std(0) + eps
    return (mean, std)


def normalize_states(states, mean, std):
    return (states - mean) / std


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = ([], [])
    ep_ret, ep_len = (0.0, 0)
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = (0.0, 0)
    lengths.append(ep_len)
    assert sum(lengths) == len(dataset["rewards"])
    return (min(returns), max(returns))


def modify_reward(
    dataset, env_name, max_episode_steps=1000, reward_scale=1.0, reward_bias=0.0
):
    if any((s in env_name for s in ("halfcheetah", "hopper", "walker2d"))):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] = dataset["rewards"] / (max_ret - min_ret)
        dataset["rewards"] = dataset["rewards"] * max_episode_steps
    if reward_scale != 1.0 or reward_bias != 0.0:
        dataset["rewards"] = dataset["rewards"] * reward_scale + reward_bias
    return dataset


def load_dataset(
    env_name,
    host_token,
    reward_transform,
    reward_scale_antmaze,
    reward_bias_antmaze,
    normalize_reward=False,
    normalize=True,
    cql_reward_scale=1.0,
    cql_reward_bias=0.0,
    with_next_action=False,
    max_episode_steps=1000,
):
    env = gym.make(env_name)
    if with_next_action:
        from runtime.next_actions import qlearning_dataset_with_next_actions

        dataset = qlearning_dataset_with_next_actions(env)
    else:
        dataset = d4rl.qlearning_dataset(env)
    dataset["rewards"] = onp.asarray(dataset["rewards"], dtype=onp.float64)
    if normalize_reward:
        dataset = modify_reward(
            dataset,
            env_name,
            max_episode_steps=max_episode_steps,
            reward_scale=cql_reward_scale,
            reward_bias=cql_reward_bias,
        )
    dataset = apply_reward_transform(
        dataset,
        env_name,
        reward_transform,
        reward_scale_antmaze,
        reward_bias_antmaze,
        legacy_print=True,
    )
    obs = onp.asarray(dataset["observations"], dtype=onp.float32)
    next_obs = onp.asarray(dataset["next_observations"], dtype=onp.float32)
    if normalize:
        mean, std = compute_mean_std(obs, eps=0.001)
        obs = normalize_states(obs, mean, std)
        next_obs = normalize_states(next_obs, mean, std)
    else:
        mean = onp.zeros(obs.shape[-1], dtype=onp.float32)
        std = onp.ones(obs.shape[-1], dtype=onp.float32)
    fields = dict(
        obs=jnp.asarray(obs),
        action=jnp.asarray(dataset["actions"], dtype=jnp.float32),
        reward=jnp.asarray(dataset["rewards"], dtype=jnp.float32),
        next_obs=jnp.asarray(next_obs),
        done=jnp.asarray(dataset["terminals"], dtype=jnp.float32),
    )
    if with_next_action:
        fields["next_action"] = jnp.asarray(dataset["next_actions"], dtype=jnp.float32)
        transitions = TransitionNA(**fields)
    else:
        transitions = Transition(**fields)
    print(f"Dataset size: {fields['obs'].shape[0]}")
    return (transitions, jnp.asarray(mean), jnp.asarray(std))


def make_eval_env(env_name, num_workers):
    _kitchen_compat.prepare_kitchen_backend(env_name)
    if any((x in env_name for x in ["maze2d", "kitchen", "antmaze"])):
        return gym.vector.SyncVectorEnv([lambda: gym.make(env_name)] * num_workers)
    return gym.vector.make(env_name, num_envs=num_workers)


def eval_env_seed(run_seed, index):
    return int((int(run_seed) * 1000003 + int(index)) % (2**31 - 1))


def eval_policy(env, rng, num_workers, act_single, obs_mean, obs_std, eval_seed=None):
    step = 0
    returned = onp.zeros(num_workers).astype(bool)
    cum_reward = onp.zeros(num_workers)
    if eval_seed is not None and hasattr(env, "seed"):
        try:
            env.seed(int(eval_seed))
        except Exception as _e:
            warnings.warn(f"could not seed evaluation env: {_e}")
    obs = env.reset()

    @jax.jit
    @jax.vmap
    def _policy_step(rng_i, obs_i):
        obs_i = (obs_i - obs_mean) / obs_std
        return jnp.nan_to_num(act_single(rng_i, obs_i))

    max_episode_steps = env.env_fns[0]().spec.max_episode_steps
    while step < max_episode_steps and (not returned.all()):
        step += 1
        rng, rng_step = jax.random.split(rng)
        rng_step = jax.random.split(rng_step, num_workers)
        action = _policy_step(rng_step, jnp.asarray(obs))
        obs, reward, done, _info = env.step(onp.asarray(action))
        cum_reward += reward * ~returned
        returned |= done
    if step >= max_episode_steps and (not returned.all()):
        warnings.warn("Maximum steps reached before all episodes terminated")
    return cum_reward


def normalized_scores(env_name, returns):
    return d4rl.get_normalized_score(env_name, returns) * 100.0


def log_eval_line(step, scores, elapsed, extra=""):
    rate = step / max(elapsed, 1e-06)
    print(
        "Step:",
        step,
        f"\t Score: {scores.mean():.2f}\t t={elapsed:.1f}s ({rate:.0f} steps/s){extra}",
    )


def save_final_npz(args, returns, scores, elapsed, steps_completed, out_dir=None):
    out_dir = out_dir or "evaluation/final_returns"
    os.makedirs(out_dir, exist_ok=True)

    def _agg(x, k):
        return {k: x, f"{k}_mean": x.mean(), f"{k}_std": x.std()}

    info = _agg(returns, "final_returns") | _agg(scores, "final_scores")
    info = info | {
        "wallclock_s": onp.float32(elapsed),
        "steps_per_s": onp.float32(args.num_updates / max(elapsed, 1e-06)),
        "steps_completed": onp.int64(steps_completed),
    }
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{args.algorithm}_{args.dataset}_s{args.seed}_{time_str}.npz"
    path = os.path.join(out_dir, filename)
    with open(path, "wb") as f:
        onp.savez_compressed(f, **info, args=asdict(args))
    print(f"[npz] wrote {path}")
    return path


def run_final_eval(
    args, rng, env, act_single, obs_mean, obs_std, elapsed, steps_completed
):
    if args.eval_final_episodes <= 0:
        print("[npz] eval_final_episodes=0 -> no NPZ written (probe run)")
        return None
    final_iters = int(onp.ceil(args.eval_final_episodes / args.eval_workers))
    print(f"Evaluating final agent for {final_iters} iterations...")
    rngs = jax.random.split(rng, final_iters)
    rets = onp.array(
        [
            eval_policy(
                env,
                r,
                args.eval_workers,
                act_single,
                obs_mean,
                obs_std,
                eval_seed=eval_env_seed(args.seed, 10**6 + i),
            )
            for i, r in enumerate(rngs)
        ]
    )
    scores = normalized_scores(args.dataset, rets)
    print(
        f"Final score: {scores.mean():.2f} +- {scores.std():.2f} over {rets.size} episodes"
    )
    return save_final_npz(args, rets, scores, elapsed, steps_completed)


from runtime.networks import (
    pytorch_init,
    uniform_init,
    orthogonal_init,
    torch_dense,
    identity,
    torch_adam,
)

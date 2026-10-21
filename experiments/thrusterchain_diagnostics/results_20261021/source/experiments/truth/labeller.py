"""Rollout truth Q^pi(s, a) for a frozen deterministic policy: batched MuJoCo sims, one jitted actor call per step.

Second piece of the D4RL-RT truth instrument (experiments/truth/README.md). Given start states (qpos, qvel)
and first actions a, a label is the discounted return of: take a, then follow pi, with discount gamma = 0.99,
until the body is unhealthy (termination) or `horizon` = 1,000 env steps after the restored state, counting
the first step. The gym time limit is ignored, as the critics' targets ignore it (qlearning_dataset drops
timeout rows). The tail cut at 1,000 steps is at most 0.99^1000 x V_max, about 0.02 raw units for these tasks.
With deterministic physics and a deterministic policy one rollout IS the value: there is no Monte Carlo
noise in the deterministic mode.

Policy (FrozenPolicy). pi(s) = max_action * tanh(MLP((float32(s) - mean) / std)), the repo's TD3+BC Actor
(algorithms/td3_bc.Actor) with its frozen params and the training-row normalization (prepared.obs_mean,
obs_std). The forward pass is one jax.jit program of FIXED shape (batch, obs_dim): XLA CPU picks different
kernels for different batch sizes, which changes float32 rounding (measured here: up to 1.5e-7 between batch
sizes), but at a fixed shape a row's output does not depend on its position or on the other rows. The
position-invariance gate (FrozenPolicy.position_invariance) re-checks that bit for bit on the actual inputs
and the labeller refuses to run without it, because chaos would otherwise turn rounding into different
labels depending on which slot a rollout ran in. Evaluation (runtime/td3_bc.evaluate_episodes) runs the
same function eagerly, so its actions can differ from these in the last float32 bit; that difference is of
the kind the jitter mode below measures, and is not a different policy.

Batched rollouts (rollouts). P = policy.batch slots (256 by default), each a bare MjSim from
state_restore.Physics. Each MjSim holds about 4 MB of mjData (the models' 1.28M-entry stack) and takes 4-18 ms
to create, so a process creates its sims once and reuses them for every shard; at P = 256 the policy call
costs about 1.2-1.6 us per env step against about 100 us of physics, and a larger P buys little. Rollouts
are queued item-major ((item 0, repeat 0), (item 0, repeat 1), ...) and fill slots first-free; a finished
slot is refilled from the queue at once. Per env step: one jitted actor call on the (P, obs_dim) observation
buffer of all slots (idle slots hold zeros; the call is skipped when every live slot is at its first step,
which uses the given action), then each live slot takes Physics.step with its action (bitwise env.step:
NormalizedBoxEnv's action map, 4 MuJoCo steps, the env's reward and termination). A restore is a full
mjData reset, so a label does not depend on the slot, the batch neighbours or what the slot ran before.
test_labeller.py checks labels against an independent direct gym rollout (env.step through the whole
wrapper chain) of the same policy: equal bit for bit, for both tasks, short and full 1,000-step rollouts.

Modes (LabelSpec.mode), each with K = repeats rollouts per item:
  deterministic  K = 1, the exact Q^pi(s, a).
  jitter         restore (qpos + jitter_scale * sd_qpos * z_q, qvel + jitter_scale * sd_qvel * z_v), z ~ N(0, I)
                 from the rollout's own stream, sd = the data SD of each coordinate (state_restore.state_sd,
                 which gives x = qpos[0] SD 0), jitter_scale 1e-4 by default. The spread over K repeats measures
                 how rough the truth is at the scale of the data (chaos a smooth critic cannot see). The
                 arithmetic runs even at jitter_scale 0, and then reproduces the deterministic labels exactly.
  noisy_target   after the given first action, a_t = clip(pi(s_t) + clip(noise_sd * z_t, +-noise_clip), +-1),
                 z_t ~ N(0, I), noise_sd 0.2 and noise_clip 0.5: TD3's target-smoothing policy pi_eps, whose
                 value Q^{pi_eps} is what the critic's target bootstraps. Reported separately from Q^pi, never
                 counted as critic error. noise_sd 0 reproduces the deterministic labels exactly.
Random streams are per rollout: numpy PCG64 seeded by SeedSequence([seed, KEY, item_id, repeat]) with KEY =
KEY_JITTER or KEY_NOISE. A label therefore depends only on (spec, item, repeat), never on the shard or the
slot schedule.

Outputs per (item, repeat): q = discounted return (float64), ret = undiscounted return, length = env steps
taken, status (0 terminated, 1 reached the horizon, 2 simulator error: q = NaN), first_reward = r(s, a).

Action gradient (central_difference_actions, action_gradient): the true grad_a Q^pi(s, a) at a = pi(s) by
central differences over a_j +- delta (delta = 0.1), 2 d_a rollouts per state. An action pushed past +-1 is
clipped (the env would clip it anyway) and the divisor is the clipped step actually taken, so g_j is the
secant over the actions really applied; the count of clipped coordinates is returned.

Shards (run_labels / load_labels). Items are cut into shards of shard_size items; each shard is one rollouts
call, written atomically as shard_NNNNN.npz plus shard_NNNNN.json (the commit marker: npz sha256, the
manifest sha256, item range, env steps, wall time). manifest.json (strict JSON, allow_nan=False) fixes the
spec, the physics identity (sources, model bytes, versions), the policy identity (params sha256 by
runtime/td3_bc._tree_hash, normalization, batch, jax version and backend, the position-invariance gate) and
the sha256 of every input array. A rerun in the same directory refuses a different manifest, skips every
shard whose json and npz verify, and recomputes the rest; shards can be split over processes with
worker/workers (shard index % workers == worker). A resumed job reproduces a fresh one bit for bit
(test_labeller.py).

Throughput (benchmark). `python -m experiments.truth.labeller benchmark ...` times the batched engine and
writes strict JSON. --policy random_host: a randomly initialised actor of the host's exact architecture from
logged start states; it falls within tens of steps, so restores weigh more than with the real host.
--policy upright (upright_actor): a PD controller that stays up for the full 1,000 steps while paying the
host actor's forward cost every step, the closer stand-in for a competent host. Measured in the build
container (1 process, 256 slots, another agent sharing the 4 CPUs, 2026-10-11; engineering timings, not
findings): Hopper 9,215 env steps/s, Walker2d 5,898 env steps/s with upright; 8,343 and 6,144 with
random_host. At those rates one critic's labels in the design (118M and 184M env steps) take about 3.6 and
8.7 process-hours.

Nothing here trains anything. No BCA band or BCA arm is involved.

JAX_PLATFORMS=cpu LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin python -m experiments.truth.labeller benchmark \
    --dataset hopper --data FILE.hdf5 --slots 256 --rollouts 512 --policy upright \
    --output runs/truth/benchmark/hopper.json
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.truth.state_restore import (  # noqa: E402
    Physics, array_sha256, observation, require, sha256_bytes, task_spec)

MANIFEST_SCHEMA = "bca-truth-labels-v1"
SHARD_SCHEMA = "bca-truth-label-shard-v1"
BENCHMARK_SCHEMA = "bca-truth-benchmark-v1"
MODES = ("deterministic", "jitter", "noisy_target")
KEY_JITTER = 1246647368  # SeedSequence keys of the per-rollout streams
KEY_NOISE = 1313821509
STATUS = {0: "terminated", 1: "horizon", 2: "simulator_error"}
OUTPUTS = dict(q="float64", ret="float64", length="int32", status="int8", first_reward="float64")
DESIGN_ENV_STEPS = dict(hopper=118e6, walker2d=184e6)  # design_d4rl_rt.json: env steps of one critic's labels


def strict_json(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False)


def canonical_sha256(value):
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


# ---------------------------------------------------------------- policy

class FrozenPolicy:
    """The frozen deterministic actor as one fixed-shape jitted program: (batch, obs_dim) float64 -> float32."""

    def __init__(self, actor, params, obs_mean, obs_std, *, batch=256):
        import jax
        import jax.numpy as jnp

        require(type(batch) is int and batch >= 1, "explicit positive batch")
        self.actor, self.params, self.batch = actor, params, batch
        self.obs_dim, self.action_dim = int(actor.obs_dim), int(actor.action_dim)
        self.max_action = float(actor.max_action)
        self.mean = np.asarray(obs_mean, np.float32).reshape(-1)
        self.std = np.asarray(obs_std, np.float32).reshape(-1)
        require(self.mean.shape == (self.obs_dim,) and self.std.shape == (self.obs_dim,), "normalization width")
        require(np.isfinite(self.mean).all() and np.all(self.std > 0), "invalid normalization")
        mean, std = jnp.asarray(self.mean), jnp.asarray(self.std)
        self._fn = jax.jit(lambda p, obs: actor.apply(p, (obs - mean) / std))
        self._jax = jax
        self._identity = None

    @classmethod
    def from_host_state(cls, models, native, obs_mean, obs_std, *, batch=256):
        """The evaluated TD3+BC policy of a restored host checkpoint: models[0] (algorithms/td3_bc.Actor) with
        native.actor.params (not the target actor, as runtime/td3_bc.evaluate_episodes) and the prepared
        training-row normalization (prepared.obs_mean, prepared.obs_std)."""
        from algorithms.td3_bc import Actor

        require(type(models[0]) is Actor, "models[0] must be the TD3+BC Actor")
        return cls(models[0], native.actor.params, obs_mean, obs_std, batch=batch)

    def __call__(self, obs):
        obs = np.asarray(obs, np.float32)
        require(obs.shape == (self.batch, self.obs_dim), "the policy takes exactly (batch, obs_dim)")
        out = np.asarray(self._fn(self.params, obs))
        require(out.shape == (self.batch, self.action_dim) and out.dtype == np.float32, "policy output shape/dtype")
        require(np.isfinite(out).all() and np.all(np.abs(out) <= self.max_action), "nonfinite or out-of-bound action")
        return out

    def actions(self, obs):
        """pi for any number of observations, through the same fixed-shape program (rows padded with zeros)."""
        obs = np.asarray(obs, np.float64)
        require(obs.ndim == 2 and obs.shape[1] == self.obs_dim, "observations must be (n, obs_dim)")
        out = np.zeros((len(obs), self.action_dim), np.float32)
        for start in range(0, len(obs), self.batch):
            chunk = obs[start:start + self.batch]
            padded = np.zeros((self.batch, self.obs_dim))
            padded[:len(chunk)] = chunk
            out[start:start + len(chunk)] = self(padded)[:len(chunk)]
        return out

    def position_invariance(self, probe):
        """Each probe row alone at several slots of a zero batch, and the rows reversed, must give the
        bitwise-same action as in place. Returns the gate record."""
        probe = np.asarray(probe, np.float64)[:self.batch]
        require(probe.ndim == 2 and len(probe) and probe.shape[1] == self.obs_dim, "nonempty probe required")
        full = np.zeros((self.batch, self.obs_dim))
        full[:len(probe)] = probe
        reference = self(full)[:len(probe)]
        diffs = []
        reversed_batch = np.zeros_like(full)
        reversed_batch[self.batch - len(probe):] = probe[::-1]
        diffs.append(np.abs(self(reversed_batch)[self.batch - len(probe):][::-1] - reference).max())
        slots = sorted({0, self.batch // 3, self.batch // 2, self.batch - 1})
        for i in range(min(len(probe), 8)):
            for slot in slots:
                alone = np.zeros_like(full)
                alone[slot] = probe[i]
                diffs.append(np.abs(self(alone)[slot] - reference[i]).max())
        worst = float(max(diffs))
        return dict(passed=worst == 0.0, max_abs_difference=worst, probe_rows=int(len(probe)),
                    slots=[int(s) for s in slots], rule="bitwise equality of a row's action across slots and batch "
                    "neighbours at the fixed batch shape")

    def identity(self):
        if self._identity is None:
            from runtime.td3_bc import _tree_hash

            jax = self._jax
            self._identity = dict(
                architecture=f"{type(self.actor).__module__}.{type(self.actor).__name__}",
                obs_dim=self.obs_dim, action_dim=self.action_dim, max_action=self.max_action,
                params_sha256=_tree_hash(self.params), obs_mean=self.mean.tolist(), obs_std=self.std.tolist(),
                normalization_sha256=array_sha256(np.concatenate([self.mean, self.std])), batch=self.batch,
                jax=jax.__version__, backend=jax.default_backend(), device=str(jax.devices()[0]),
                forward="jit(actor.apply(params, (float32(obs) - mean) / std)) at fixed shape (batch, obs_dim)")
        return dict(self._identity)


def random_host_networks(task, seed):
    """Randomly initialised actor and twin critic of the host's exact architecture (algorithms/td3_bc.initialize):
    (actor module, critic module, native AgentTrainState). For tests and benchmarks only."""
    from algorithms import td3_bc

    task = task_spec(task) if isinstance(task, str) else task
    args = td3_bc.Args(seed=seed, dataset=task.dataset)
    _, native, (actor, critic) = td3_bc.initialize(args, task.obs_dim, task.action_dim, 1.0)
    return actor, critic, native


def upright_actor(task, host_weight=1e-6):
    """Benchmark/test policy with the Actor interface: a fixed joint-PD controller that keeps Hopper and Walker2d
    upright for 1,000 steps from reset states, plus host_weight x a randomly initialised host-architecture
    Actor evaluated on the same input, so every call pays the host's full forward cost. Use with obs_mean 0
    and obs_std 1 (the PD gains act on raw observations). Not a trained policy; never a source of findings."""
    import flax.linen as nn
    import jax.numpy as jnp

    from algorithms.td3_bc import Actor

    spec = task_spec(task) if isinstance(task, str) else task
    pitch_gain = 0.0 if spec.name == "hopper" else -1.0

    class UprightActor(nn.Module):
        obs_dim: int
        action_dim: int
        max_action: float

        @nn.compact
        def __call__(self, obs):
            host = Actor(self.obs_dim, self.action_dim, self.max_action)(obs)
            angles, rates = obs[..., 2:spec.nq - 1], obs[..., spec.nq + 2:]
            pre = -0.5 * angles - 0.05 * rates + pitch_gain * obs[..., 1:2]
            return self.max_action * jnp.tanh(pre + host_weight * host)

    return UprightActor(spec.obs_dim, spec.action_dim, 1.0)


# ---------------------------------------------------------------- the batched engine

@dataclass(frozen=True)
class LabelSpec:
    dataset: str
    mode: str = "deterministic"
    repeats: int = 1
    gamma: float = 0.99
    horizon: int = 1000
    seed: int = 0
    jitter_scale: float = 1e-4
    noise_sd: float = 0.2
    noise_clip: float = 0.5
    shard_size: int = 4096

    def __post_init__(self):
        task_spec(self.dataset)
        require(self.mode in MODES, f"mode must be one of {MODES}")
        require(type(self.repeats) is int and self.repeats >= 1, "repeats must be a positive integer")
        require(self.mode != "deterministic" or self.repeats == 1, "deterministic mode has exactly one repeat")
        require(type(self.horizon) is int and 1 <= self.horizon <= 1000, "horizon in [1, 1000] env steps")
        require(type(self.seed) is int and 0 <= self.seed < 2**63, "explicit nonnegative seed")
        require(type(self.shard_size) is int and self.shard_size >= 1, "positive shard size")
        for name in ("gamma", "jitter_scale", "noise_sd", "noise_clip"):
            v = getattr(self, name)
            require(isinstance(v, float) and np.isfinite(v) and v >= 0, name + " must be a finite float >= 0")
        require(self.gamma <= 1.0, "gamma <= 1")


def _stream(seed, key, item_id, repeat):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, key, int(item_id), int(repeat)])))


def check_items(task, items):
    """Validate an item set: qpos (n, nq), qvel (n, nv) float64, action (n, d_a) float32 in [-1, 1], unique item_id."""
    n = len(items["item_id"])
    out = dict(qpos=np.asarray(items["qpos"], np.float64), qvel=np.asarray(items["qvel"], np.float64),
               action=np.asarray(items["action"]), item_id=np.asarray(items["item_id"]))
    require(n >= 1 and out["item_id"].dtype == np.int64 and out["item_id"].shape == (n,), "int64 item ids required")
    require(len(np.unique(out["item_id"])) == n and np.all(out["item_id"] >= 0), "unique nonnegative item ids")
    require(out["qpos"].shape == (n, task.nq) and out["qvel"].shape == (n, task.nv), "state arrays shape")
    require(out["action"].dtype == np.float32 and out["action"].shape == (n, task.action_dim), "float32 actions")
    require(np.isfinite(out["qpos"]).all() and np.isfinite(out["qvel"]).all(), "nonfinite start states")
    require(np.isfinite(out["action"]).all() and np.all(np.abs(out["action"]) <= 1.0), "actions must lie in [-1, 1]")
    return out


def rollouts(physics, policy, spec, items, *, jitter_sd=None, sims=None, timing=None):
    """Labels of every (item, repeat), shape (n, K), plus env steps. One jitted actor call per env step at most."""
    task = physics.task
    require(task.dataset == task_spec(spec.dataset).dataset, "spec and physics name different tasks")
    require(policy.obs_dim == task.obs_dim and policy.action_dim == task.action_dim, "policy shape differs from task")
    items = check_items(task, items)
    n, K, H, P = len(items["item_id"]), spec.repeats, spec.horizon, policy.batch
    jitter = spec.mode == "jitter"
    noisy = spec.mode == "noisy_target"
    if jitter:
        require(jitter_sd is not None, "jitter mode needs the data SD of qpos and qvel")
        sd_q, sd_v = (np.asarray(x, np.float64) for x in jitter_sd)
        require(sd_q.shape == (task.nq,) and sd_v.shape == (task.nv,) and np.all(sd_q >= 0) and np.all(sd_v >= 0),
                "jitter SDs must be nonnegative (nq,) and (nv,)")
        step_q, step_v = spec.jitter_scale * sd_q, spec.jitter_scale * sd_v
    total = n * K
    out = dict(q=np.full((n, K), np.nan), ret=np.full((n, K), np.nan), length=np.zeros((n, K), np.int32),
               status=np.full((n, K), -1, np.int8), first_reward=np.full((n, K), np.nan))
    width = min(P, total)
    sims = list(sims or [])
    while len(sims) < width:
        sims.append(physics.new_sim())
    obs = np.zeros((P, task.obs_dim))
    slot_item = np.full(P, -1, np.int64)
    slot_rep = np.zeros(P, np.int64)
    t = np.zeros(P, np.int64)
    G, R, disc = np.zeros(P), np.zeros(P), np.ones(P)
    eps = [None] * P
    clock = dict(policy=0.0, physics=0.0, restore=0.0, policy_calls=0, restores=0, env_steps=0)
    queue = 0

    def load(s):
        nonlocal queue
        i, k = divmod(queue, K)
        queue += 1
        qpos, qvel = items["qpos"][i], items["qvel"][i]
        if jitter:
            z = _stream(spec.seed, KEY_JITTER, items["item_id"][i], k).standard_normal(task.nq + task.nv)
            qpos, qvel = qpos + step_q * z[:task.nq], qvel + step_v * z[task.nq:]
        if noisy:
            z = _stream(spec.seed, KEY_NOISE, items["item_id"][i], k).standard_normal((H, task.action_dim))
            eps[s] = np.clip(spec.noise_sd * z, -spec.noise_clip, spec.noise_clip).astype(np.float32)
        started = time.perf_counter()
        physics.restore(sims[s], qpos, qvel)
        obs[s] = physics.observation(sims[s])
        clock["restore"] += time.perf_counter() - started
        clock["restores"] += 1
        slot_item[s], slot_rep[s], t[s], G[s], R[s], disc[s] = i, k, 0, 0.0, 0.0, 1.0

    def finish(s, status):
        i, k = slot_item[s], slot_rep[s]
        out["q"][i, k] = G[s] if status != 2 else np.nan
        out["ret"][i, k] = R[s] if status != 2 else np.nan
        out["length"][i, k], out["status"][i, k] = t[s], status
        if queue < total:
            load(s)
        else:
            slot_item[s] = -1
            obs[s] = 0.0

    for s in range(width):
        load(s)
    while True:
        live = np.flatnonzero(slot_item >= 0)
        if not len(live):
            break
        acts = None
        if np.any(t[live] > 0):
            started = time.perf_counter()
            acts = policy(obs)
            clock["policy"] += time.perf_counter() - started
            clock["policy_calls"] += 1
        started = time.perf_counter()
        for s in live:
            i = slot_item[s]
            if t[s] == 0:
                a = items["action"][i]
            else:
                a = acts[s]
                if noisy:
                    a = np.clip(a + eps[s][t[s]], -1.0, 1.0)
            try:
                reward, done = physics.step(sims[s], a)
            except physics.simulator_error:  # MuJoCo instability: label NaN, status 2, the slot is reloaded
                t[s] += 1
                finish(s, 2)
                continue
            if t[s] == 0:
                out["first_reward"][i, slot_rep[s]] = reward
            G[s] = G[s] + disc[s] * reward
            R[s] = R[s] + reward
            disc[s] = disc[s] * spec.gamma
            t[s] += 1
            clock["env_steps"] += 1
            if done:
                finish(s, 0)
            elif t[s] >= H:
                finish(s, 1)
            else:
                obs[s] = physics.observation(sims[s])
        clock["physics"] += time.perf_counter() - started
    require(np.all(out["status"] >= 0), "every rollout must finish")
    if timing is not None:
        timing.update(clock)
    return out


# ---------------------------------------------------------------- items helpers

def policy_actions(physics, policy, qpos, qvel):
    """pi(s) at start states (the observation the restored sim would show), through the fixed-shape program."""
    return policy.actions(observation(physics.task, qpos, qvel))


def central_difference_actions(pi_actions, delta=0.1):
    """(plus, minus, step, clipped): plus[i, j] = clip(pi_i + delta e_j, -1, 1), minus likewise (float32,
    shape (n, d, d)); step[i, j] = plus[i, j, j] - minus[i, j, j] in float64; clipped counts coordinates
    whose step was shortened by the +-1 bound."""
    pi = np.asarray(pi_actions, np.float32)
    require(pi.ndim == 2 and np.all(np.abs(pi) <= 1.0), "(n, d) actions in [-1, 1] required")
    require(isinstance(delta, float) and 0 < delta <= 1, "delta in (0, 1]")
    d = pi.shape[1]
    e = np.eye(d, dtype=np.float32) * np.float32(delta)
    plus = np.clip(pi[:, None, :] + e[None], -1.0, 1.0).astype(np.float32)
    minus = np.clip(pi[:, None, :] - e[None], -1.0, 1.0).astype(np.float32)
    j = np.arange(d)
    step = plus[:, j, j].astype(np.float64) - minus[:, j, j].astype(np.float64)
    clipped = int(np.sum(step < 2 * np.float64(np.float32(delta)) * (1 - 1e-6)))
    return plus, minus, step, clipped


def action_gradient(q_plus, q_minus, step):
    """grad_a Q by central differences: (Q(s, a + h e_j) - Q(s, a - h e_j)) / step_j, shape (n, d)."""
    q_plus, q_minus, step = (np.asarray(x, np.float64) for x in (q_plus, q_minus, step))
    require(q_plus.shape == q_minus.shape == step.shape and np.all(step > 0), "aligned (n, d) arrays required")
    return (q_plus - q_minus) / step


def gradient_items(physics, policy, qpos, qvel, item_ids, delta=0.1):
    """Items for the action gradient at pi(s): 2 d_a rollouts per state, ordered [+e_0, ..., +e_{d-1}, -e_0, ...].
    item_ids (n,) are the state ids; item k*2d + m gets id item_ids[k] * 2d + m."""
    qpos, qvel = np.asarray(qpos, np.float64), np.asarray(qvel, np.float64)
    pi = policy_actions(physics, policy, qpos, qvel)
    plus, minus, step, clipped = central_difference_actions(pi, delta)
    n, d = pi.shape
    actions = np.concatenate([plus, minus], axis=1).reshape(n * 2 * d, d)
    ids = (np.asarray(item_ids, np.int64)[:, None] * 2 * d + np.arange(2 * d)[None]).reshape(-1)
    items = dict(qpos=np.repeat(qpos, 2 * d, axis=0), qvel=np.repeat(qvel, 2 * d, axis=0), action=actions,
                 item_id=ids)
    return items, dict(pi=pi, step=step, clipped=clipped, delta=delta)


def gradient_from_labels(q, n_states, action_dim, step):
    """Split (n * 2d,) deterministic labels from gradient_items into plus/minus and difference them."""
    q = np.asarray(q, np.float64).reshape(n_states, 2 * action_dim)
    return action_gradient(q[:, :action_dim], q[:, action_dim:], step)


# ---------------------------------------------------------------- shards

def _write_atomic(path, data, mode="wb"):
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with open(tmp, mode) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _npz_bytes(arrays):
    import io

    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def build_manifest(physics, policy, spec, items, jitter_sd, invariance):
    n = len(items["item_id"])
    inputs = {k: array_sha256(items[k]) for k in ("qpos", "qvel", "action", "item_id")}
    if jitter_sd is not None:
        inputs["jitter_sd"] = array_sha256(np.concatenate([np.asarray(x, np.float64) for x in jitter_sd]))
    return dict(
        schema=MANIFEST_SCHEMA, spec=asdict(spec), physics=physics.identity(), policy=policy.identity(),
        position_invariance=invariance, n_items=n, shard_size=spec.shard_size,
        n_shards=-(-n // spec.shard_size), inputs=inputs, outputs=dict(OUTPUTS, item_id="int64"),
        definition=("q = sum_t gamma^t r_t over: first action a, then the mode's policy, until termination or "
                    "horizon env steps after the restored state; time limit ignored"),
        status_codes={str(k): v for k, v in STATUS.items()})


def _shard_paths(out_dir, index):
    return out_dir / f"shard_{index:05d}.npz", out_dir / f"shard_{index:05d}.json"


def _verify_shard(out_dir, index, manifest_sha, ids, repeats):
    npz_path, json_path = _shard_paths(out_dir, index)
    if not (npz_path.is_file() and json_path.is_file()):
        return None
    try:
        record = json.loads(json_path.read_text(encoding="utf8"))
        payload = npz_path.read_bytes()
        if (record.get("schema") != SHARD_SCHEMA or record.get("manifest_sha256") != manifest_sha
                or record.get("npz_sha256") != sha256_bytes(payload)):
            return None
        with np.load(npz_path) as npz:
            arrays = {k: npz[k] for k in npz.files}
        if set(arrays) != set(OUTPUTS) | {"item_id"} or not np.array_equal(arrays["item_id"], ids):
            return None
        if any(arrays[k].shape != (len(ids), repeats) or arrays[k].dtype != np.dtype(v) for k, v in OUTPUTS.items()):
            return None
        return arrays, record
    except (ValueError, OSError, KeyError):
        return None


def prepare_directory(out_dir, manifest):
    """Create manifest.json, or verify that the existing one is this job. Returns the manifest sha256."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.json"
    document = dict(manifest=manifest, manifest_sha256=canonical_sha256(manifest))
    text = strict_json(document)
    if not path.exists():
        tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
        tmp.write_text(text, encoding="utf8")
        try:
            os.link(tmp, path)  # atomic create; another worker may have won the race
        except FileExistsError:
            pass
        except OSError:  # no hard links on this filesystem (e.g. some WSL /mnt drives): exclusive create instead
            try:
                with open(path, "x", encoding="utf8") as handle:
                    handle.write(text)
            except FileExistsError:
                pass
        finally:
            tmp.unlink()
    existing = json.loads(path.read_text(encoding="utf8"))
    if existing != json.loads(text):
        raise ValueError(f"{path} describes a different label job; refusing to mix labels (use a new directory)")
    return document["manifest_sha256"]


def run_labels(out_dir, physics, policy, spec, items, *, jitter_sd=None, worker=0, workers=1, max_shards=None,
               log=None):
    """Label items into resumable shards under out_dir. Returns the list of shard records (computed or verified)."""
    items = check_items(physics.task, items)
    require(type(workers) is int and workers >= 1 and type(worker) is int and 0 <= worker < workers, "worker split")
    n = len(items["item_id"])
    probe = observation(physics.task, items["qpos"][:policy.batch], items["qvel"][:policy.batch])
    invariance = policy.position_invariance(probe)
    require(invariance["passed"], "the jitted actor is not position-invariant at this batch shape; labels would "
            "depend on the slot schedule: " + json.dumps(invariance))
    manifest = build_manifest(physics, policy, spec, items, jitter_sd, invariance)
    out_dir = Path(out_dir)
    manifest_sha = prepare_directory(out_dir, manifest)
    records, computed = [], 0
    sims = [physics.new_sim() for _ in range(min(policy.batch, spec.shard_size * spec.repeats))]
    for index in range(manifest["n_shards"]):
        if index % workers != worker:
            continue
        lo, hi = index * spec.shard_size, min(n, (index + 1) * spec.shard_size)
        verified = _verify_shard(out_dir, index, manifest_sha, items["item_id"][lo:hi], spec.repeats)
        if verified is not None:
            records.append(dict(verified[1], resumed=True))
            continue
        if max_shards is not None and computed >= max_shards:
            break
        part = {k: v[lo:hi] for k, v in items.items()}
        timing, started = {}, time.perf_counter()
        labels = rollouts(physics, policy, spec, part, jitter_sd=jitter_sd, sims=sims, timing=timing)
        seconds = time.perf_counter() - started
        arrays = dict({k: labels[k].astype(v) for k, v in OUTPUTS.items()}, item_id=items["item_id"][lo:hi])
        payload = _npz_bytes(arrays)
        npz_path, json_path = _shard_paths(out_dir, index)
        _write_atomic(npz_path, payload)
        record = dict(schema=SHARD_SCHEMA, manifest_sha256=manifest_sha, shard=index, items=[lo, hi],
                      rollouts=int((hi - lo) * spec.repeats), npz_sha256=sha256_bytes(payload),
                      env_steps=int(timing["env_steps"]), seconds=seconds,
                      env_steps_per_second=timing["env_steps"] / max(seconds, 1e-9),
                      status_counts={STATUS[c]: int((labels["status"] == c).sum()) for c in STATUS},
                      timing={k: timing[k] for k in ("policy", "physics", "restore", "policy_calls", "restores")},
                      utc=datetime.now(timezone.utc).isoformat())
        _write_atomic(json_path, strict_json(record).encode())
        records.append(dict(record, resumed=False))
        computed += 1
        if log is not None:
            log(f"shard {index}: {record['rollouts']} rollouts, {record['env_steps']} env steps, "
                f"{record['env_steps_per_second']:.0f} steps/s")
    return records


def load_labels(out_dir):
    """Every shard of a complete job, verified and concatenated in item order: (arrays, manifest)."""
    out_dir = Path(out_dir)
    document = json.loads((out_dir / "manifest.json").read_text(encoding="utf8"))
    manifest = document["manifest"]
    require(document["manifest_sha256"] == canonical_sha256(manifest), "manifest digest mismatch")
    parts = []
    for index in range(manifest["n_shards"]):
        npz_path, _ = _shard_paths(out_dir, index)
        require(npz_path.is_file(), f"shard {index} missing; the job is incomplete")
        with np.load(npz_path) as npz:
            ids = npz["item_id"]
        verified = _verify_shard(out_dir, index, document["manifest_sha256"], ids, manifest["spec"]["repeats"])
        require(verified is not None, f"shard {index} does not verify")
        parts.append(verified[0])
    arrays = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    require(len(arrays["item_id"]) == manifest["n_items"], "item count differs from the manifest")
    require(array_sha256(arrays["item_id"]) == manifest["inputs"]["item_id"], "item ids differ from the manifest")
    return arrays, manifest


# ---------------------------------------------------------------- benchmark

def benchmark(dataset, data_file, *, slots, rollouts_count, seed, horizon=1000, policy_kind="random_host"):
    """Time the engine on logged start states, plus physics-only and policy-only rates.

    policy_kind 'random_host': randomly initialised host actor with the data's normalization (short rollouts:
    it falls within tens of steps). 'upright': upright_actor, which pays the host actor's forward cost every
    step and stays upright, so rollouts run to the 1,000-step horizon as the real host's mostly do."""
    from runtime.common import compute_mean_std
    from experiments.truth.state_restore import converted_rows, load_raw, logged_states

    physics = Physics(dataset)
    try:
        task = physics.task
        raw = load_raw(physics, data_file)
        converted, rows, _ = converted_rows(raw)
        states = logged_states(task, raw, rows)
        kept = np.flatnonzero(states.kept)
        g = np.random.default_rng(seed)
        pick = np.sort(g.choice(kept, size=min(rollouts_count, len(kept)), replace=False))
        if len(pick) < rollouts_count:
            pick = np.resize(pick, rollouts_count)
        if policy_kind == "random_host":
            mean, std = compute_mean_std(np.asarray(converted["observations"], np.float32), eps=0.001)
            actor, _, native = random_host_networks(task, seed)
            policy = FrozenPolicy(actor, native.actor.params, mean, std, batch=slots)
            description = "random-init host actor (algorithms/td3_bc.initialize), NOT a trained policy"
        else:
            import jax

            require(policy_kind == "upright", "policy_kind must be 'random_host' or 'upright'")
            actor = upright_actor(task)
            params = actor.init(jax.random.PRNGKey(seed), np.zeros((1, task.obs_dim), np.float32))
            policy = FrozenPolicy(actor, params, np.zeros(task.obs_dim), np.ones(task.obs_dim), batch=slots)
            description = ("upright_actor: joint-PD controller + 1e-6 x random-init host-architecture Actor (full "
                           "host forward cost per step), NOT a trained policy")
            # start from reset states, where the PD controller is known to stay upright
            starts = []
            for i in range(rollouts_count):
                physics.env.seed(int(seed * 1000003 + i) % (2**31 - 1))
                physics.env.reset()
                starts.append((physics.base.sim.data.qpos.copy(), physics.base.sim.data.qvel.copy()))
            states = None
        if states is not None:
            items = dict(qpos=states.qpos[pick], qvel=states.qvel[pick],
                         action=np.asarray(converted["actions"])[pick].astype(np.float32),
                         item_id=np.arange(rollouts_count, dtype=np.int64))
        else:
            qpos, qvel = np.stack([q for q, _ in starts]), np.stack([v for _, v in starts])
            items = dict(qpos=qpos, qvel=qvel, action=policy_actions(physics, policy, qpos, qvel),
                         item_id=np.arange(rollouts_count, dtype=np.int64))
        spec = LabelSpec(task.dataset, horizon=horizon, shard_size=rollouts_count)
        policy(np.zeros((slots, task.obs_dim)))  # compile outside the timed region
        started = time.perf_counter()
        sims = [physics.new_sim() for _ in range(slots)]  # once per process; run_labels reuses them across shards
        sim_creation = time.perf_counter() - started
        timing, started = {}, time.perf_counter()
        labels = rollouts(physics, policy, spec, items, sims=sims, timing=timing)
        seconds = time.perf_counter() - started
        # physics only: the same sims stepped with fixed random actions, no policy, no restores
        for s, i in zip(sims, range(slots)):
            physics.restore(s, items["qpos"][i % rollouts_count], items["qvel"][i % rollouts_count])
        acts = g.uniform(-1, 1, size=(20, slots, task.action_dim)).astype(np.float32)
        t0 = time.perf_counter()
        for step in range(20):
            for j, s in enumerate(sims):
                physics.step(s, acts[step, j])
        physics_rate = 20 * slots / (time.perf_counter() - t0)
        obs = np.zeros((slots, task.obs_dim))
        t0 = time.perf_counter()
        for _ in range(50):
            policy(obs)
        policy_ms = 1e3 * (time.perf_counter() - t0) / 50
        rate = timing["env_steps"] / seconds
        result = dict(
            schema=BENCHMARK_SCHEMA, dataset=task.dataset, data_file=str(Path(data_file).resolve()),
            data_sha256=sha256_bytes(Path(data_file).read_bytes()), synthetic_data=_is_synthetic(data_file),
            slots=slots, rollouts=rollouts_count, horizon=horizon, seed=seed, policy_kind=policy_kind,
            policy=description,
            env_steps=int(timing["env_steps"]), seconds=seconds, env_steps_per_second=rate,
            sim_creation_seconds=sim_creation,
            mean_rollout_length=float(labels["length"].mean()),
            status_counts={STATUS[c]: int((labels["status"] == c).sum()) for c in STATUS},
            time_shares={k: timing[k] / seconds for k in ("policy", "physics", "restore")},
            policy_calls=int(timing["policy_calls"]), policy_call_ms=policy_ms,
            physics_only_env_steps_per_second=physics_rate,
            long_rollout_estimate_env_steps_per_second=1.0 / (1.0 / physics_rate + policy_ms / 1e3 / slots),
            design_env_steps=DESIGN_ENV_STEPS[task.name], cpu_count=os.cpu_count(),
            omp_num_threads=os.environ.get("OMP_NUM_THREADS"), physics=physics.identity(),
            policy_identity=policy.identity(), utc=datetime.now(timezone.utc).isoformat(),
            note="engineering timing on this machine; not a finding")
        for key, value in (("measured", rate),
                           ("long_rollout_estimate", result["long_rollout_estimate_env_steps_per_second"])):
            result[f"design_label_process_hours_at_{key}_rate"] = DESIGN_ENV_STEPS[task.name] / value / 3600
        return result
    finally:
        physics.close()


def _is_synthetic(path):
    try:
        from experiments.truth.synthetic import is_synthetic

        return bool(is_synthetic(path))
    except Exception:
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("benchmark", help="time the batched engine (random-init actor) and write strict JSON")
    b.add_argument("--dataset", required=True)
    b.add_argument("--data", required=True, type=Path, help="D4RL-format HDF5 file for start states")
    b.add_argument("--slots", type=int, default=256)
    b.add_argument("--rollouts", type=int, default=512)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--policy", choices=("random_host", "upright"), default="random_host")
    b.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "benchmark":
        result = benchmark(args.dataset, args.data, slots=args.slots, rollouts_count=args.rollouts, seed=args.seed,
                           policy_kind=args.policy)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf8") as handle:
            handle.write(strict_json(result))
        print(strict_json({k: result[k] for k in ("dataset", "policy_kind", "slots", "env_steps", "seconds",
                                                   "env_steps_per_second",
                                                   "mean_rollout_length", "policy_call_ms",
                                                   "physics_only_env_steps_per_second",
                                                   "long_rollout_estimate_env_steps_per_second")}))


if __name__ == "__main__":
    main()

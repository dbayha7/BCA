"""Two Q^pi evaluators (protocol 4.5).

An anchor is an exact simulator state and time rebuilt by replaying a recorder prefix in the stall world (Part 0,
L2): {"reset_seed", "current": (lo, hi), "prefix_actions": float32 (t, 4)}. Q^pi(s, a) takes the forced first
command a at that state, then follows the policy to the episode's end: 100 - t steps, discount 0.98, no bootstrap
after step 99, rewards summed from exponent 0. Both worlds' values start from the same stall-world state.

- `env_values`: the environment-based evaluator. It rebuilds each anchor through the canonical environment
  (seeded reset, then env.step over the prefix in `stall3`), switches the environment's physics to the evaluated
  world for the continuation (asserting the world dict), and reads observations from the environment.
- `raw_values`: a raw loop over canonical `dynamics()` with explicit float64 state, discount and horizon. It draws
  the start with draw_starts from PCG64(reset_seed), replays the prefix itself and builds its own observations and
  remaining-time input.

Only the raw actor forward pass is shared, on the main evaluator's batch layout: at each offset >= 1 both call
`forward(obs_batch)` on the active rows of the full anchor x slot layout, in row order. Each evaluator applies the
continuation's clip (all four commands to +-KNEE when `clip_all`) and the float32 deployment cast itself.
"""
import numpy as np

import worlds
from core import require
from knee_region import KNEE

GAMMA, HORIZON = 0.98, 100
PREFIX_WORLD = "stall3"


def _check(anchors, first_actions):
    first_actions = np.asarray(first_actions)
    require(first_actions.dtype == np.float32 and first_actions.ndim == 3 and first_actions.shape[-1] == 4
            and first_actions.shape[0] == len(anchors), "First actions must be float32 (anchors, slots, 4)")
    times = np.asarray([len(a["prefix_actions"]) for a in anchors])
    require(bool(np.all((times >= 0) & (times < HORIZON))), "Anchor times must be in [0, 100)")
    for a in anchors:
        p = np.asarray(a["prefix_actions"])
        require(p.dtype == np.float32 and p.ndim == 2 and p.shape[1] == 4, "Prefix actions must be float32 (t, 4)")
    wide = first_actions.astype(np.float64)
    require(bool(np.all(np.isfinite(wide))) and bool(np.all(np.abs(wide) <= 1.0)),
            "Forced first actions must be finite and within [-1, 1]; no hidden clipping")
    return first_actions, times


def _deploy(raw, clip_all):
    raw = np.asarray(raw)
    require(raw.dtype == np.float32 and raw.ndim == 2 and raw.shape[1] == 4, "Actor output must be float32 (rows, 4)")
    a = raw.astype(np.float64)
    if clip_all:
        a = np.clip(a, -KNEE, KNEE)
    require(bool(np.all(np.isfinite(a))) and bool(np.all(np.abs(a) <= 1.0)), "Deployed actor command out of range")
    return a.astype(np.float32)


def raw_anchor_state(anchor):
    """Exact float64 anchor state by a raw replay: draw_starts from PCG64(reset_seed), then dynamics() in stall3."""
    env = worlds.load_environment()
    law = env.canonical.start_law(env.START_LAWS["recorded"], {"current": tuple(anchor["current"])})
    x = env.draw_starts(1, np.random.Generator(np.random.PCG64(int(anchor["reset_seed"]))), law)[0]
    physics = worlds.physics(PREFIX_WORLD)
    for a in np.asarray(anchor["prefix_actions"]):
        x, _ = env.dynamics(x, np.clip(a.astype(np.float64), -1.0, 1.0), physics)
    return x


def raw_values(anchors, first_actions, forward, world, *, clip_all=True):
    """Raw dynamics() loop. Returns (values (anchors, slots), forward calls, receipt)."""
    first_actions, times = _check(anchors, first_actions)
    states = np.stack([raw_anchor_state(a) for a in anchors])
    return raw_values_from_states(states, times, first_actions, forward, world, clip_all=clip_all)


def raw_values_from_states(states, times, first_actions, forward, world, *, clip_all=True):
    """The raw loop from exact float64 states and times (used by 4.8 on the prior study's saved banks, whose
    continuation is the unclipped actor: clip_all=False). Rows are states repeated per slot, as in the recorded
    `continuation_values` (reference.py 259-272), so the actor sees the recorded batch at every offset.
    Returns (values, forward calls, receipt of the physics actually used)."""
    states, times = np.asarray(states), np.asarray(times)
    first_actions = np.asarray(first_actions)
    require(states.dtype == np.float64 and states.ndim == 2 and states.shape[1] == 13, "Float64 (n, 13) states")
    wide = np.asarray(first_actions).astype(np.float64)
    require(bool(np.all(np.isfinite(wide))) and bool(np.all(np.abs(wide) <= 1.0)), "Forced first actions out of range")
    require(times.shape == (len(states),) and times.dtype.kind in "iu" and bool(np.all((times >= 0) & (times < HORIZON))),
            "Times must be integers in [0, 100)")
    require(first_actions.dtype == np.float32 and first_actions.shape[:1] == (len(states),) and first_actions.shape[-1] == 4,
            "First actions must be float32 (n, slots, 4)")
    env = worlds.load_environment()
    physics = worlds.physics(world)
    n, slots = first_actions.shape[:2]
    remaining = np.repeat(HORIZON - times, slots)
    x = np.repeat(states, slots, axis=0).copy()
    values = np.zeros(n * slots, np.float64)
    first = first_actions.reshape(-1, 4)
    calls = 0
    for offset in range(int(remaining.max())):
        active = np.flatnonzero(offset < remaining)
        if offset == 0:
            a = first[active]
        else:
            step_index = HORIZON - remaining[active] + offset
            obs = np.concatenate([x[active], ((HORIZON - step_index) / HORIZON)[:, None]], axis=1).astype(np.float32)
            a = _deploy(forward(obs), clip_all)
            calls += 1
        nxt, reward = env.dynamics(x[active], np.clip(a.astype(np.float64), -1.0, 1.0), physics)
        values[active] += GAMMA ** offset * reward
        x[active] = nxt
    return values.reshape(n, slots), calls, _receipt("raw_dynamics_loop", physics, world, clip_all, GAMMA)


def _receipt(evaluator, continuation_physics, world, clip_all, gamma, prefix_physics=None):
    out = {"evaluator": evaluator, "continuation": worlds.assert_world(continuation_physics, world),
           "clip": "all four commands to +-KNEE" if clip_all else "none (deployed command, |a| <= 1 asserted)",
           "gamma": gamma, "horizon": HORIZON, "bootstrap": "none after step 99"}
    if prefix_physics is not None:
        out["prefix"] = worlds.assert_world(prefix_physics, "stall3")
    return out


def _rebuilt_env(anchor, world):
    env = worlds.environment("stall3")            # literal here; the raw loop uses PREFIX_WORLD
    obs, _ = env.reset(seed=int(anchor["reset_seed"]), options={"start": {"current": tuple(anchor["current"])}})
    for a in np.asarray(anchor["prefix_actions"]):
        obs, _, terminated, truncated, _ = env.step(a)
        require(not (terminated or truncated), "Prefix reached the horizon")
    worlds.assert_world(env.physics, "stall3")    # the prefix world is the environment's own, certified
    if world != "stall3":
        env.physics = worlds.physics(world)        # continuation world; the anchor state stays the stall-world state
    worlds.assert_world(env.physics, world)
    return env, obs


def _env_deploy(raw, clip_all):
    """The environment evaluator's own deployment (independent of the raw loop's _deploy)."""
    out = np.array(raw, dtype=np.float32, copy=True)
    require(out.ndim == 2 and out.shape[1] == 4 and np.asarray(raw).dtype == np.float32, "Actor output must be float32 (rows, 4)")
    if clip_all:
        knee = np.float32(worlds.KNEE)
        out = np.minimum(np.maximum(out, -knee), knee)
    require(bool(np.isfinite(out).all()) and bool((np.abs(out) <= np.float32(1)).all()), "Deployed command out of range")
    return out


def env_values(anchors, first_actions, forward, world, *, clip_all=True):
    """Environment-based evaluator on the same layout. Returns (values (anchors, slots), forward calls, receipt)."""
    first_actions, _ = _check(anchors, first_actions)
    n, slots = first_actions.shape[:2]
    pairs = [_rebuilt_env(anchors[i], world) for i in range(n) for _ in range(slots)]
    envs = [p[0] for p in pairs]
    obs = np.stack([p[1] for p in pairs]).astype(np.float32)
    values = np.zeros(n * slots, np.float64)
    discount = np.ones(n * slots, np.float64)
    done = np.zeros(n * slots, bool)
    first = first_actions.reshape(-1, 4)
    calls, offset = 0, 0
    while not np.all(done):
        active = np.flatnonzero(~done)
        if offset == 0:
            a = first[active]
        else:
            a = _env_deploy(forward(obs[active]), clip_all)
            calls += 1
        for j, i in enumerate(active):
            o, r, terminated, truncated, info = envs[i].step(a[j])
            values[i] += discount[i] * r
            discount[i] *= envs[i].physics.gamma
            obs[i] = o
            done[i] = terminated or truncated
        offset += 1
    gamma = envs[0].physics.gamma
    used = envs[0].physics
    for env in envs:
        require(env.physics == used, "Environments ran different physics")
        env.close()
    return values.reshape(n, slots), calls, _receipt("environment", used, world, clip_all, gamma,
                                                     prefix_physics=worlds.physics("stall3"))


def cross_check(raw, env, rtol=1e-9):
    """4.5 agreement: |raw - env| <= rtol x max(1, |value|) everywhere. Returns (ok, largest scaled deviation)."""
    raw, env = np.asarray(raw, np.float64), np.asarray(env, np.float64)
    scaled = np.abs(raw - env) / np.maximum(1.0, np.abs(env))
    return bool(np.all(scaled <= rtol)), float(scaled.max())

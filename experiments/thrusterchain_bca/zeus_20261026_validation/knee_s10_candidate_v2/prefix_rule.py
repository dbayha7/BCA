"""Candidate knee_s10_v2: the anchor prefix rule, the one change from knee_s10_v1 (G2_DEV_RULE_v2.md).

Every prefix draw is mapped onto CUTOFF's retained support for the load band of the state the command is given in,
with the same five uniforms per step the recorder would use (u, v, w1, w2, w3):
  high load (band >= 1): a0 = -KNEE + (KNEE - 0.6) v, uniform on [-0.8, -0.6]
  low load  (band == 0): a0 = -KNEE + 2 KNEE v,       uniform on [-0.8, 0.8]
  a1..a3 = -0.7 + 1.4 w_k, the recorder's own side rule.
The executed float32 command is clipped into that support, so float32 rounding can never put it past the knee.
No prefix command is past the knee, so the stall never acts during a prefix and the trajectory is the same in both
worlds; build_anchor runs the prefix in both and refuses unless the observations and states agree bit for bit.
"""
import numpy as np

from core import require
from knee_region import KNEE, load_band, load_g, past_knee_a0
import recorder_wide
import stage1_eval as E
import worlds

HIGH_UPPER = float(np.float32(-0.6))          # the retained high-load support's upper edge (as in_cutoff_support)
SIDE_RANGE = recorder_wide.SIDE_RANGE         # 0.7, never past the knee
RULE = {"name": "knee_s10_v2_prefix_in_support", "uniforms_per_step": ["u", "v", "w1", "w2", "w3"],
        "rule": "if load_band(g) >= 1: a0 = -KNEE + (KNEE - 0.6) v; else a0 = -KNEE + 2 KNEE v; a_k = -0.7 + 1.4 w_k",
        "support": "CUTOFF's retained support per load band: high a0 in [-KNEE, -0.6], low a0 in [-KNEE, KNEE], |a_k| <= 0.7",
        "u": "read but unused (the window probability has no role inside the support)", "load": recorder_wide.BEHAVIOR["load"]}


def band_of(observation):
    return int(load_band(load_g(np.asarray(observation, np.float32)[None]))[0])


def requested_command(observation, uniforms):
    """(requested float64 (4,), band) for one observation and the step's five uniforms."""
    u, v, w1, w2, w3 = (float(x) for x in uniforms)
    band = band_of(observation)
    a0 = -KNEE + (HIGH_UPPER + KNEE) * v if band >= 1 else -KNEE + 2.0 * KNEE * v
    requested = np.array([a0, -SIDE_RANGE + 2 * SIDE_RANGE * w1, -SIDE_RANGE + 2 * SIDE_RANGE * w2,
                          -SIDE_RANGE + 2 * SIDE_RANGE * w3], np.float64)
    return requested, band


def executed_command(observation, uniforms):
    """The float32 command the prefix executes: the request cast to float32 and clipped into the band's support."""
    requested, band = requested_command(observation, uniforms)
    cmd = requested.astype(np.float32)
    hi = np.float32(HIGH_UPPER) if band >= 1 else np.float32(KNEE)
    cmd[0] = np.minimum(np.maximum(cmd[0], np.float32(-KNEE)), hi)
    cmd[1:] = np.clip(cmd[1:], np.float32(-SIDE_RANGE), np.float32(SIDE_RANGE))
    return cmd


def in_retained_support(obs, a):
    """Per row: is command a inside CUTOFF's retained data for the load band of obs? The formal driver's
    in_cutoff_support, repeated here so the prefix rule can check itself; test_v2 asserts the two agree."""
    high = load_band(load_g(obs)) >= 1
    a0, side = a[:, 0].astype(np.float64), np.abs(a[:, 1:].astype(np.float64))
    hi = np.where(high, HIGH_UPPER, KNEE)
    return (a0 >= -KNEE) & (a0 <= hi) & np.all(side <= SIDE_RANGE + 1e-7, axis=1)


def check_prefix(obs_before, prefix):
    """P1 on one prefix: every executed command inside the support of its own state's band and none past the knee."""
    obs_before, prefix = np.asarray(obs_before, np.float32), np.asarray(prefix, np.float32)
    require(obs_before.shape[:1] == prefix.shape[:1] and prefix.shape[1:] == (4,), "Prefix and observations misaligned")
    if len(prefix) == 0:
        return
    require(bool(np.all(in_retained_support(obs_before, prefix))), "A prefix command is outside the retained support")
    require(not bool(np.any(past_knee_a0(prefix)[0])), "A prefix command is past the knee")


def build_anchor(name, item, stream):
    """A development anchor: the bank item and prefix uniforms of the test-only stream (as stage1_eval's smoke path),
    the v2 prefix run in BOTH worlds, which must agree bit for bit (P1)."""
    spec = E.stream_item(name, item, stream)
    t = int(spec["t"])
    uniforms = np.random.Generator(np.random.PCG64(E._smoke_sequence(stream, "prefix", name, item))).random((t, 5))
    runs = {}
    for world in ("stall3", "nostall"):
        env = worlds.environment(world)
        try:
            obs, _ = env.reset(seed=spec["reset_seed"], options={"start": {"current": spec["current"]}})
            prefix, trace = np.zeros((t, 4), np.float32), [np.asarray(obs, np.float32).copy()]
            for step in range(t):
                prefix[step] = executed_command(obs, uniforms[step])
                obs, _, terminated, truncated, _ = env.step(prefix[step])
                require(not (terminated or truncated), "Prefix reached the horizon")
                trace.append(np.asarray(obs, np.float32).copy())
            runs[world] = {"prefix": prefix, "trace": np.stack(trace), "state": np.asarray(env.state, np.float64).copy()}
        finally:
            env.close()
    a, b = runs["stall3"], runs["nostall"]
    require(np.array_equal(a["prefix"], b["prefix"]) and np.array_equal(a["trace"], b["trace"])
            and np.array_equal(a["state"], b["state"]), "The prefix differs between the worlds (P1)")
    check_prefix(a["trace"][:-1], a["prefix"])
    return {"bank": name, "item": int(item), "t": t, "reset_seed": spec["reset_seed"], "current": tuple(spec["current"]),
            "prefix_actions": a["prefix"], "obs": a["trace"][-1], "state": a["state"], "both_worlds_identical": True,
            "smoke": True}

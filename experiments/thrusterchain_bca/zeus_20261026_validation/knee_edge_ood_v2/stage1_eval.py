"""Stage 1 evaluation computations (protocol Stage 1 "Evaluation" and "R0"), on a policy object with the interface of
rerun1's FrozenPolicy: act(obs) -> float32 (n, 4) raw deployed commands, q(obs, actions) -> (n, 2) twin Q, and
assert_unchanged(). No model is loaded here, so every function runs on Windows with a stand-in actor.

- Free episodes: a raw float64 loop over canonical dynamics() from a bank start (draw_starts from PCG64(reset_seed),
  t = 0), 100 steps, discount 0.98 from exponent 0. Variants: pi (the raw command, |a| <= 1 asserted), pi_clip (a0
  clipped to +-KNEE) and pi_clipall (all four clipped). The executed float32 commands, observations and exact states
  are kept, so policy-visited states are rebuilt exactly (the prefix replay of the free rollouts).
- Step rates on E2 free steps of pi: R_deep-, any-depth W by load band, W+, a1..a3 past the knee and in (0.7, KNEE].
- Q probes at the Stage 1 anchors (the A1 probe builder and evaluator): the gated slots S9 + s05..p10 at a1..a3 = 0
  with the clipped (primary) and unclipped (reported) own continuation and Q1/Q2; Q_nostall on the same 14-slot array
  under a stall-trained policy's clipped continuation (S9, s05, s10, s20 are used); the reported actor-own a1..a3 set plus the actor's own action;
  Q1 at a1 = +-KNEE and +-0.69999999 with a0 at the actor's value (Part 6.10, a2 = a3 = 0).
- The rate of target-smoothed actions in W, as an exact expectation over the TD3 clipped Gaussian (no random draws).
"""
import hashlib
import math

import numpy as np

import probes as P
import qpi
import recorder
import region_independent as second
import seeds as S
import worlds
from core import require
from knee_region import GATED_PAST_KNEE, KNEE, S9, SLOTS, in_deep_minus, load_band, load_g, past_knee_a0

GAMMA, HORIZON = 0.98, 100
VARIANTS = ("pi", "clip", "clipall")
GATED_SLOTS = tuple(S9) + tuple(GATED_PAST_KNEE)
NOSTALL_SLOTS = GATED_SLOTS                 # the same probe array as the stall values: identical actor batches for the 1e-9 check
VISITED_SLOTS = ("edge_knee",) + tuple(GATED_PAST_KNEE)
A1_VALUES = (KNEE, -KNEE, -SLOTS["edge_700"], SLOTS["edge_700"])          # +KNEE, -KNEE, +0.69999999, -0.69999999
VISITED_T, VISITED_PER_T = (0, 25, 50), 16


SMOKE_KEYS = {"reset": 9901, "prefix": 9903, "mixture": 9904}


def _smoke_sequence(smoke_seed, key, name, item):
    salt = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big")
    return np.random.SeedSequence(int(smoke_seed), spawn_key=(SMOKE_KEYS[key], salt, int(item)), pool_size=4)


def stream_item(name, item, smoke_seed=None):
    """A bank item's draw rule: the sealed rule (seeds.bank_item) for research, or, for the smoke test only, a
    test-only rule with the same structure (count, law, anchor times) drawn from smoke_seed, so a smoke run never
    touches a sealed evaluation stream."""
    if smoke_seed is None:
        return S.bank_item(name, item)
    spec = S.BANKS[name]
    require(0 <= int(item) < spec["count"], "Item outside the bank")
    reset = int(_smoke_sequence(smoke_seed, "reset", name, item).generate_state(1, dtype=np.uint64)[0]) & ((1 << 63) - 1)
    if spec["law"] == "e2":
        current = S.E2_CURRENT
    else:
        mix = np.random.Generator(np.random.PCG64(_smoke_sequence(smoke_seed, "mixture", name, item))).random()
        current = recorder.CURRENT_RANGES[int(mix < recorder.DEEP_PROBABILITY)]
    out = {"bank": name, "item": int(item), "reset_seed": reset, "current": tuple(current), "smoke": True}
    if spec["kind"] == "anchors":
        out["t"] = spec["t_schedule"][int(item) // (spec["count"] // len(spec["t_schedule"]))]
    return out


def bank_states(name, smoke_seed=None):
    """Exact float64 start states of a starts bank, by the raw path (the anchors' raw replay with t = 0)."""
    rows = []
    for i in range(S.BANKS[name]["count"]):
        spec = stream_item(name, i, smoke_seed)
        rows.append(qpi.raw_anchor_state({"reset_seed": spec["reset_seed"], "current": spec["current"],
                                          "prefix_actions": np.zeros((0, 4), np.float32)}))
    return np.stack(rows)


def build_anchor(name, item, smoke_seed=None):
    """Research: step04.build_anchor exactly (the sealed rule). Smoke: the same recorder-prefix replay in stall3 from
    the test-only item and uniforms."""
    import step04
    if smoke_seed is None:
        return step04.build_anchor(name, item)
    spec = stream_item(name, item, smoke_seed)
    t = spec["t"]
    uniforms = np.random.Generator(np.random.PCG64(_smoke_sequence(smoke_seed, "prefix", name, item))).random((t, 5))
    env = worlds.environment("stall3")
    try:
        obs, _ = env.reset(seed=spec["reset_seed"], options={"start": {"current": spec["current"]}})
        prefix = np.zeros((t, 4), np.float32)
        for step in range(t):
            prefix[step] = recorder.requested_command(obs, uniforms[step])[0].astype(np.float32)
            obs, _, terminated, truncated, _ = env.step(prefix[step])
            require(not (terminated or truncated), "Prefix reached the horizon")
        return {"bank": name, "item": item, "t": t, "reset_seed": spec["reset_seed"], "current": spec["current"],
                "prefix_actions": prefix, "state": env.state, "obs": np.asarray(obs, np.float32), "smoke": True}
    finally:
        env.close()


def deploy(raw, variant):
    raw = np.asarray(raw)
    require(raw.dtype == np.float32 and raw.ndim == 2 and raw.shape[1] == 4, "Actor output must be float32 (rows, 4)")
    require(bool(np.all(np.isfinite(raw))) and bool(np.all(np.abs(raw.astype(np.float64)) <= 1.0)), "Actor command out of range")
    out = raw.copy()
    knee = np.float32(KNEE)
    if variant == "clip":
        out[:, 0] = np.minimum(np.maximum(out[:, 0], -knee), knee)
    elif variant == "clipall":
        out = np.minimum(np.maximum(out, -knee), knee)
    else:
        require(variant == "pi", "Unknown variant")
    return out


def rollouts(states, forward, world, variant, budget):
    """Free episodes from exact start states. Returns returns (n,), executed (n, 100, 4) float32, obs (n, 100, 14)
    float32 and states (n, 100, 13) float64 (the state before each step), plus the physics receipt."""
    states = np.asarray(states)
    require(states.dtype == np.float64 and states.ndim == 2 and states.shape[1] == 13, "Float64 (n, 13) start states")
    n = len(states)
    budget.charge_raw(n * HORIZON)
    env = worlds.load_environment()
    physics = worlds.physics(world)
    x = states.copy()
    returns = np.zeros(n, np.float64)
    executed = np.zeros((n, HORIZON, 4), np.float32)
    obs_all = np.zeros((n, HORIZON, 14), np.float32)
    xs = np.zeros((n, HORIZON, 13), np.float64)
    for t in range(HORIZON):
        obs = np.concatenate([x, np.full((n, 1), (HORIZON - t) / HORIZON)], axis=1).astype(np.float32)
        a = deploy(forward(obs), variant)
        xs[:, t], obs_all[:, t], executed[:, t] = x, obs, a
        x, reward = env.dynamics(x, np.clip(a.astype(np.float64), -1.0, 1.0), physics)
        returns += GAMMA ** t * np.asarray(reward, np.float64)
    require(bool(np.all(np.isfinite(returns))), "Nonfinite free-episode return")
    return {"returns": returns, "executed": executed, "obs": obs_all, "states": xs,
            "receipt": {"world": worlds.assert_world(physics, world), "variant": variant, "episodes": n,
                        "gamma": GAMMA, "horizon": HORIZON}}


def classify(actions):
    """W mask, side and R_deep- of executed commands, by both region implementations row for row (protocol 4.2:
    one disagreement stops the unit)."""
    a = np.asarray(actions)
    mask, side = past_knee_a0(a)
    mask2, side2 = second.withheld_region(a)
    deep = in_deep_minus(a)
    require(np.array_equal(mask, mask2) and np.array_equal(side, np.asarray(side2, side.dtype))
            and np.array_equal(deep, second.in_deep_minus(a)), "Region implementations disagree on an evaluated action")
    return mask, side, deep


def a123_past(actions):
    """Per row: any of a1..a3 past the knee, each component tested by both implementations (moved to index 0)."""
    a = np.asarray(actions)
    out = np.zeros(a.shape[:-1], bool)
    for k in (1, 2, 3):
        moved = a[..., [k, 0, 2, 3] if k == 1 else ([k, 1, 0, 3] if k == 2 else [k, 1, 2, 0])]
        out |= classify(np.ascontiguousarray(moved))[0]
    return out


def step_rates(executed, obs):
    """Rates over all free steps (E2 bank, pi). Executed commands are float32 (n, 100, 4)."""
    a = np.asarray(executed, np.float32).reshape(-1, 4)
    o = np.asarray(obs, np.float32).reshape(-1, 14)
    band = load_band(load_g(o))
    require(np.array_equal(band, second.load_band(o)), "Region implementations disagree on a load band")
    mask, side, deep = classify(a)
    side_abs = np.abs(a[:, 1:].astype(np.float64))
    out = {"steps": int(len(a)), "r_deep_minus": float(np.mean(deep)), "w_plus": float(np.mean(side == 1)),
           "w_any_depth": float(np.mean(mask)),
           "a123_past_knee": float(np.mean(a123_past(a))),
           "a123_in_0p7_knee": float(np.mean(np.any((side_abs > 0.7) & (side_abs <= KNEE), axis=1))),
           "by_band": {}}
    for b, name in enumerate(("low", "mid", "deep")):
        sel = band == b
        out["by_band"][name] = {"steps": int(sel.sum()),
                                "w_any_depth": float(np.mean(mask[sel])) if sel.any() else None,
                                "r_deep_minus": float(np.mean(deep[sel])) if sel.any() else None,
                                "w_minus": float(np.mean(side[sel] == -1)) if sel.any() else None}
    return out


def visited_indices(seed):
    """The sealed visited-state rule: per t in {0, 25, 50}, 16 of the 128 E2 bank episodes without replacement from
    PCG64(SeedSequence(seed, spawn_key=(1905, t)))."""
    return {t: np.sort(np.random.Generator(np.random.PCG64(np.random.SeedSequence(int(seed), spawn_key=(1905, t))))
                       .choice(128, VISITED_PER_T, replace=False)) for t in VISITED_T}


def visited_states(rollout, seed):
    """(states (48, 13) float64, obs (48, 14) float32, times (48,)) at the selected episodes and times."""
    picks = visited_indices(seed)
    rows = [(int(e), t) for t in VISITED_T for e in picks[t]]
    states = np.stack([rollout["states"][e, t] for e, t in rows])
    obs = np.stack([rollout["obs"][e, t] for e, t in rows])
    return states, obs, np.asarray([t for _, t in rows], np.int64), rows


def critic(policy, obs, probes_array):
    """Q1, Q2 and the twin minimum at every (row, slot) of a probe array."""
    n, slots = probes_array.shape[:2]
    q = np.asarray(policy.q(np.repeat(np.asarray(obs, np.float32), slots, axis=0), probes_array.reshape(-1, 4)))
    require(q.shape == (n * slots, 2) and np.isfinite(q).all(), "Expected finite twin Q")
    q = q.astype(np.float64).reshape(n, slots, 2)
    return q[..., 0], q[..., 1], q.min(axis=2)


def a1_probes(own):
    """(n, 4, 4) float32: a0 at the actor's value, a1 in (+KNEE, -KNEE, +0.69999999, -0.69999999), a2 = a3 = 0."""
    own = np.asarray(own, np.float32)
    out = np.zeros((len(own), len(A1_VALUES), 4), np.float32)
    out[..., 0] = own[:, None, 0]
    out[..., 1] = np.asarray(A1_VALUES, np.float32)[None, :]
    require(bool(np.all(np.abs(out.astype(np.float64)) <= 1.0)), "a1 probe out of range")
    return out


def anchor_values(policy, anchors, world, budget, *, stall_trained):
    """Every Q probe of one policy at a list of anchors ({reset_seed, current, prefix_actions, state, obs})."""
    n = len(anchors)
    obs = np.stack([np.asarray(a["obs"], np.float32) for a in anchors])
    times = np.asarray([len(a["prefix_actions"]) for a in anchors], np.int64)
    rows = [{"reset_seed": a["reset_seed"], "current": tuple(a["current"]),
             "prefix_actions": np.asarray(a["prefix_actions"], np.float32)} for a in anchors]
    gated = P.build(n, GATED_SLOTS)
    out = {"times": times, "gated_slots": list(GATED_SLOTS)}
    budget.charge(times, gated.shape[1])
    out["qpi_clip"], calls_c, out["receipt_clip"] = qpi.raw_values(rows, gated, policy.act, world, clip_all=True)
    budget.charge(times, gated.shape[1])
    out["qpi_unclipped"], calls_u, out["receipt_unclipped"] = qpi.raw_values(rows, gated, policy.act, world, clip_all=False)
    out["q1"], out["q2"], out["qmin"] = critic(policy, obs, gated)
    out["valid"] = np.all(np.isfinite(out["qpi_clip"]), axis=1)
    if stall_trained:
        ns = P.build(n, NOSTALL_SLOTS)
        budget.charge(times, ns.shape[1])
        out["qpi_nostall"], _, out["receipt_nostall"] = qpi.raw_values(rows, ns, policy.act, "nostall", clip_all=True)
        out["nostall_slots"] = list(NOSTALL_SLOTS)
        out["valid"] &= np.all(np.isfinite(out["qpi_nostall"]), axis=1)
    own = policy.act(obs)
    classify(own)
    excluded = a123_past(own)
    own_probes = np.concatenate([P.build(n, GATED_SLOTS, a123=own[:, 1:], kind="actor_own"), own[:, None, :]], axis=1)
    budget.charge(times, own_probes.shape[1])
    own_q, _, _ = qpi.raw_values(rows, own_probes, policy.act, world, clip_all=True)
    own_q1 = critic(policy, obs, own_probes)[0]
    own_q[excluded], own_q1[excluded] = np.nan, np.nan            # Part 0: excluded from the reports, and counted
    out["own_qpi_clip"], out["own_q1"] = own_q, own_q1
    out["own_slots"] = list(GATED_SLOTS) + ["actor_own_action"]
    out["own_a123_past_knee"] = excluded
    out["own_excluded"] = int(excluded.sum())
    out["a1_q1"] = critic(policy, obs, a1_probes(own))[0]
    out["a1_values"] = [float(v) for v in A1_VALUES]
    out["actor_action"] = own
    out["forward_calls"] = int(calls_c + calls_u)
    policy.assert_unchanged()
    return out


def stall_equals_nostall(anchor_out, rtol=1e-9):
    """V1: under the clipped continuation Q_nostall equals Q_stall on the supported S9 slots, on the same probe array.
    Returns (ok, largest scaled deviation)."""
    names, ns_names = list(anchor_out["gated_slots"]), list(anchor_out["nostall_slots"])
    q, ns = np.asarray(anchor_out["qpi_clip"], np.float64), np.asarray(anchor_out["qpi_nostall"], np.float64)
    return qpi.cross_check(q[:, [names.index(n) for n in S9]], ns[:, [ns_names.index(n) for n in S9]], rtol)


def target_smoothing_rate(mu_a0, noise, noise_clip):
    """Exact mean probability that a TD3 target action clip(mu + clip(N(0, noise^2), -c, c), -1, 1) has |a0| > KNEE,
    over the rows' deterministic target-actor a0 values mu."""
    mu = np.asarray(mu_a0, np.float64)
    phi = np.frompyfunc(lambda z: 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))), 1, 1)
    x, y = KNEE - mu, -KNEE - mu
    up = np.where(x >= noise_clip, 0.0, np.where(x < -noise_clip, 1.0, 1.0 - phi(x / noise).astype(np.float64)))
    down = np.where(y <= -noise_clip, 0.0, np.where(y > noise_clip, 1.0, phi(y / noise).astype(np.float64)))
    return float(np.mean(up + down))


def world_shift(norm_stall, norm_nostall):
    """Per-input standardized mean difference between the two worlds' training inputs (their CUTOFF n RANDOM
    normalization fixtures), pooled standard deviation."""
    m1, m2 = np.asarray(norm_stall["mean"], np.float64), np.asarray(norm_nostall["mean"], np.float64)
    s = np.sqrt((np.asarray(norm_stall["std"], np.float64) ** 2 + np.asarray(norm_nostall["std"], np.float64) ** 2) / 2)
    return ((m1 - m2) / np.where(s > 0, s, 1.0)).tolist()


def standardized_shift(obs, mean, std):
    """Per-input standardized mean difference of policy-visited observations against the run's own training inputs
    (its normalization statistics); reported as the visited-state shift."""
    obs = np.asarray(obs, np.float64).reshape(-1, 14)
    std = np.where(np.asarray(std, np.float64) > 0, np.asarray(std, np.float64), 1.0)
    return ((obs.mean(axis=0) - np.asarray(mean, np.float64)) / std).tolist()

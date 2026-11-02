"""New goal/hazard task over an injected, unchanged 13-state ThrusterChain plant.

Importing this module never imports or calls the physical simulator. The sphere
intersection rule is an explicit discrete-task edge convention, not a claim
about an unobserved continuous trajectory between plant steps.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

STATE_DIM, ACTION_DIM, HORIZON = 13, 4, 100
GAMMA = 0.98


def state_vector(value):
    x = np.asarray(value)
    if x.shape != (STATE_DIM,) or x.dtype.kind not in "fiu" or not np.all(np.isfinite(x)):
        raise ValueError("State must contain exactly 13 finite real coordinates")
    return np.array(x, dtype=np.float64, copy=True)


def time_index(value, *, allow_terminal=False):
    if isinstance(value, (bool, np.bool_)) or int(value) != value:
        raise ValueError("Time must be an integer")
    t = int(value)
    if not 0 <= t <= (HORIZON if allow_terminal else HORIZON - 1):
        raise ValueError("Time is outside the finite horizon")
    return t


def deployed_action(value):
    a = np.asarray(value)
    if a.shape != (ACTION_DIM,) or a.dtype.kind not in "fiu" or not np.all(np.isfinite(a)):
        raise ValueError("Action must contain exactly four finite real coordinates")
    return np.clip(a.astype(np.float32), -1.0, 1.0)


def observe(state, t):
    x = state_vector(state)
    t = time_index(t, allow_terminal=True)
    return np.concatenate((x.astype(np.float32), np.array([(HORIZON - t) / HORIZON], np.float32)))


def validate_sphere(region):
    c = np.asarray(region["center"], dtype=np.float64)
    r = float(region["radius"])
    if c.shape != (10,) or not np.all(np.isfinite(c)) or not np.isfinite(r) or r <= 0:
        raise ValueError("A physical region needs ten finite coordinates and a positive radius")
    return c, r


def validate_region(region):
    if region.get("kind", "sphere") == "sphere":
        return validate_sphere(region)
    if region.get("kind") != "capsule":
        raise ValueError("Unknown region kind")
    a, b = np.asarray(region["start"], np.float64), np.asarray(region["end"], np.float64)
    r = float(region["radius"])
    if a.shape != (10,) or b.shape != (10,) or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or not np.isfinite(r) or r <= 0 or np.array_equal(a, b):
        raise ValueError("Invalid physical connector capsule")
    return a, b, r


def contains(states, region):
    x = np.asarray(states, dtype=np.float64)
    if x.shape[-1] != STATE_DIM or not np.all(np.isfinite(x)):
        raise ValueError("Region query requires finite full states")
    if region.get("kind", "sphere") == "sphere":
        c, r = validate_sphere(region)
        return np.sum((x[..., :10] - c) ** 2, axis=-1) <= r * r
    a, b, r = validate_region(region)
    d = b - a
    f = np.clip(np.sum((x[..., :10] - a) * d, axis=-1) / float(d @ d), 0, 1)
    return np.sum((x[..., :10] - a - f[..., None] * d) ** 2, axis=-1) <= r * r


def _segment_distance_squared(p1, q1, p2, q2):
    """Closest points on two closed finite segments, including degenerate edges."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    if a == 0 and e == 0:
        return float(r @ r)
    if a == 0:
        s, t = 0.0, float(np.clip(f / e, 0, 1))
    else:
        c = float(d1 @ r)
        if e == 0:
            t, s = 0.0, float(np.clip(-c / a, 0, 1))
        else:
            b, denom = float(d1 @ d2), a * e - float(d1 @ d2) ** 2
            s = float(np.clip((b * f - c * e) / denom, 0, 1)) if denom > 0 else 0.0
            t = (b * s + f) / e
            if t < 0:
                t, s = 0.0, float(np.clip(-c / a, 0, 1))
            elif t > 1:
                t, s = 1.0, float(np.clip((b - c) / a, 0, 1))
    delta = p1 + s * d1 - p2 - t * d2
    return float(delta @ delta)


def edge_intersects(states, next_states, region):
    """Closed source/next/segment intersection, including tangency and zero edges."""
    x, y = np.asarray(states, np.float64), np.asarray(next_states, np.float64)
    if x.shape != y.shape or x.shape[-1] != STATE_DIM or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Edge queries require equally shaped finite full states")
    if region.get("kind", "sphere") == "capsule":
        a, b, r = validate_region(region)
        values = [_segment_distance_squared(p[:10], q[:10], a, b) <= r * r
                  for p, q in zip(x.reshape(-1, STATE_DIM), y.reshape(-1, STATE_DIM))]
        return np.asarray(values, bool).reshape(x.shape[:-1])
    c, r = validate_sphere(region)
    a, delta = x[..., :10] - c, y[..., :10] - x[..., :10]
    denom = np.sum(delta * delta, axis=-1)
    fraction = np.divide(-np.sum(a * delta, axis=-1), denom,
                         out=np.zeros_like(denom), where=denom > 0)
    fraction = np.clip(fraction, 0.0, 1.0)
    nearest = a + fraction[..., None] * delta
    return np.sum(nearest * nearest, axis=-1) <= r * r


@dataclass(frozen=True)
class Snapshot:
    """Physical state/time plus absorbing episode status, including crossed edges."""
    state: np.ndarray
    time: int
    terminated: bool = False
    terminal_reason: str | None = None

    def __post_init__(self):
        x = state_vector(self.state)
        x.setflags(write=False)
        object.__setattr__(self, "state", x)
        object.__setattr__(self, "time", time_index(self.time, allow_terminal=True))
        if not isinstance(self.terminated, (bool, np.bool_)):
            raise ValueError("Snapshot requires an explicit boolean terminal status")
        if self.terminated != (self.terminal_reason is not None):
            raise ValueError("Terminal status and reason disagree")
        if self.terminal_reason not in (None, "goal", "danger", "horizon"):
            raise ValueError("Unknown terminal reason")
        if self.time == HORIZON and not self.terminated:
            raise ValueError("A horizon snapshot cannot be live")


class Task:
    def __init__(self, plant_step, geometry, horizon=HORIZON):
        if horizon != HORIZON:
            raise ValueError("This task is frozen to a 100-step finite horizon")
        self.plant_step = plant_step
        self.geometry = geometry
        for name in ("goal", "danger", "connector"):
            validate_region(geometry[name])

    def restore(self, state, t, *, terminated, terminal_reason=None):
        """Require saved status: a danger crossing may end outside its sphere."""
        return Snapshot(state, t, terminated, terminal_reason)

    def step_snapshot(self, snapshot, action):
        if not isinstance(snapshot, Snapshot) or snapshot.terminated:
            raise ValueError("Cannot resume a terminal or invalid snapshot")
        nxt, reward, done, info = self.step(snapshot.state, snapshot.time, action)
        return Snapshot(nxt, info["t"], done, info["terminal_reason"]), reward, info

    def observe(self, state, t):
        return observe(state, t)

    def step(self, state, t, action):
        """Primitive for known-live inputs; callers must retain the returned done.

        An edge-crossing failure can end outside the hazard. Use step_snapshot
        and restore with explicit terminal status for saved/resumed episodes.
        """
        x, t, a = state_vector(state), time_index(t), deployed_action(action)
        if bool(contains(x, self.geometry["goal"])) or bool(contains(x, self.geometry["danger"])):
            raise ValueError("Cannot step an already terminal goal or danger state")
        result = self.plant_step(x, a)
        nxt = state_vector(result[0] if isinstance(result, tuple) else result)
        danger = bool(edge_intersects(x, nxt, self.geometry["danger"]))
        # Danger wins on an edge meeting both regions; this is frozen semantics.
        goal = bool(contains(nxt, self.geometry["goal"])) and not danger
        terminal = danger or goal or t + 1 == HORIZON
        reward = -1.0 - 0.01 * float(np.sum(a.astype(np.float64) ** 2)) + 100.0 * goal - 100.0 * danger
        info = {"state": nxt.copy(), "t": t + 1, "goal": goal, "danger": danger,
                "horizon": t + 1 == HORIZON, "command": a.copy(),
                "terminal_reason": "danger" if danger else "goal" if goal else "horizon" if terminal else None}
        return nxt, reward, bool(terminal), info


def tracking_controller(A, B, recipe):
    """Fixed finite-horizon linear tracking controller, clipped at the safe knee.

    This computes no plant transition. Its target/weights/initial state are
    frozen recipe inputs, never learned from a policy or BCA outcome.
    """
    A, B = np.asarray(A, np.float64), np.asarray(B, np.float64)
    if A.shape != (13, 13) or B.shape != (13, 4):
        raise ValueError("Controller matrices must be the full 13-state plant")
    initial = state_vector(recipe["initial_state"])
    duration = int(recipe["tracking_duration"])
    if not 1 <= duration <= HORIZON:
        raise ValueError("Invalid tracking duration")
    Q = np.diag([float(recipe["physical_weight"])] * 10 + [0.0] * 3)
    R = np.eye(4) * float(recipe["action_weight"])
    if recipe["physical_weight"] <= 0 or recipe["action_weight"] <= 0:
        raise ValueError("Tracking weights must be positive")
    P, p = Q * float(recipe["terminal_weight_multiplier"]), np.zeros(13)
    gains, offsets = [None] * HORIZON, [None] * HORIZON
    for t in range(HORIZON - 1, -1, -1):
        target = np.zeros(13)
        target[0:10:2] = initial[0:10:2] * max(0.0, 1.0 - t / duration)
        H = R + B.T @ P @ B
        K, k = np.linalg.solve(H, B.T @ P @ A), np.linalg.solve(H, B.T @ p)
        gains[t], offsets[t] = K, k
        p = -Q @ target + A.T @ p - A.T @ P @ B @ k
        P = Q + A.T @ P @ A - A.T @ P @ B @ K
    knee = float(np.float32(0.8))

    def control(state, t):
        x, t = state_vector(state), time_index(t)
        return np.clip(-gains[t] @ x - offsets[t], -knee, knee).astype(np.float32)
    return control


def construct_geometry(plant_step, controller, recipe):
    """Exactly one nominal route and one frozen excursion, no candidate search.

    Each plant call must be wrapped by the caller's global setup counter. A
    failed separation/geometry check raises; no radius/band fallback is allowed.
    """
    b, c, goal_time = int(recipe["b_time"]), int(recipe["c_time"]), int(recipe["goal_time"])
    if not 0 < b < c < goal_time < HORIZON:
        raise ValueError("B, C and goal reference must be ordered interior route times")
    for key in ("danger_radius_fraction", "danger_radius_max", "minimum_region_radius", "goal_radius", "connector_radius_fraction"):
        if not np.isfinite(float(recipe[key])) or float(recipe[key]) <= 0:
            raise ValueError("Invalid frozen geometry parameter " + key)
    if not 0 < float(recipe["connector_radius_fraction"]) < float(recipe["connector_trim_fraction"]) < 0.5:
        raise ValueError("Connector radius must leave B and C outside its trimmed capsule")
    if not 0 < float(recipe["danger_radius_fraction"]) < 1:
        raise ValueError("Danger must stay separated from the nominal route")
    x = state_vector(recipe["initial_state"])
    route = [x.copy()]
    for t in range(HORIZON):
        out = plant_step(x, deployed_action(controller(x, t)))
        x = state_vector(out[0] if isinstance(out, tuple) else out)
        route.append(x.copy())
    witness = [route[b].copy()]
    x = route[b].copy()
    for t in range(b, c):
        out = plant_step(x, deployed_action(-controller(x, t)))
        x = state_vector(out[0] if isinstance(out, tuple) else out)
        witness.append(x.copy())
    physical_route = np.asarray(route)[:, :10]
    center = x[:10]
    # Distance to every nominal edge, not just vertices, protects the safe route.
    a, d = physical_route[:-1] - center, np.diff(physical_route, axis=0)
    den = np.sum(d * d, axis=1)
    f = np.clip(np.divide(-np.sum(a * d, axis=1), den, out=np.zeros_like(den), where=den > 0), 0, 1)
    separation = float(np.min(np.linalg.norm(a + f[:, None] * d, axis=1)))
    radius = min(float(recipe["danger_radius_max"]), float(recipe["danger_radius_fraction"]) * separation)
    trim = float(recipe["connector_trim_fraction"])
    if not 0 < float(recipe["connector_radius_fraction"]) < trim < 0.5:
        raise ValueError("Connector radius must leave B and C outside its trimmed capsule")
    connector_delta = route[c][:10] - route[b][:10]
    connector_radius = float(recipe["connector_radius_fraction"]) * float(np.linalg.norm(route[c][:10] - route[b][:10]))
    if not np.isfinite(radius) or radius < float(recipe["minimum_region_radius"]) or connector_radius < float(recipe["minimum_region_radius"]):
        raise ValueError("Frozen geometry is degenerate; record failed setup without tuning")
    geometry = {
        "goal": {"center": route[goal_time][:10].tolist(), "radius": float(recipe["goal_radius"])},
        "danger": {"center": center.tolist(), "radius": radius},
        "connector": {"kind": "capsule", "start": (route[b][:10] + trim * connector_delta).tolist(),
                      "end": (route[c][:10] - trim * connector_delta).tolist(), "radius": connector_radius},
        "A": route[0].tolist(), "B": route[b].tolist(), "C": route[c].tolist(),
        "D": route[goal_time].tolist(),
        "b_time": b, "c_time": c, "nominal_route": np.asarray(route).tolist(),
        "excursion": np.asarray(witness).tolist(), "danger_route_separation": separation,
        "edge_semantics": "closed straight segment in ten physical coordinates",
        "recipe": dict(recipe),
    }
    for key in ("goal", "danger", "connector"):
        validate_region(geometry[key])
    if np.any(contains(np.asarray(route[:c + 1]), geometry["goal"])):
        raise ValueError("Nominal route reaches goal before the retained C endpoint")
    if np.linalg.norm(np.asarray(geometry["goal"]["center"]) - center) <= radius + float(recipe["goal_radius"]):
        raise ValueError("Danger and goal overlap; frozen setup fails")
    if bool(contains(np.asarray(route[b]), geometry["connector"])) or bool(contains(np.asarray(route[c]), geometry["connector"])):
        raise ValueError("B and C must remain outside the missing connector")
    return geometry

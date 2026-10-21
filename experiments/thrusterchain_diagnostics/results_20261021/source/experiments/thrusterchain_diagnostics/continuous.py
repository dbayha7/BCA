"""Portable continuous ThrusterChain diagnostics with one finite-horizon objective.

The physical plant below is a frozen, read-only export from the existing
``experiments.shift.thruster_chain.matrices()``.  It was obtained by importing
the original normally, including its strict dependency checks.  This module
does not import, patch, or circumvent that package.  ``verify_legacy_parity``
accepts an already normally imported original for a direct numerical check.

The diagnostic objective is sum_{k=0}^{H-t-1} gamma**k r_{t+k}, with no
terminal reward or bootstrap.  Defaults are H=100 and gamma=1.  The original
plant's gamma=.98/continuing truth conventions are not inherited silently.
Observations contain the 13 physical coordinates, the fixed episode regime
(0=stall, 1=monotone), and the remaining fraction (H-t)/H, in that order.
Dynamics and truth use float64; optional float32 deployment is explicit.
"""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.linalg import qr
from scipy.optimize import lsq_linear

DS, DX, DA, OBS_DIM = 10, 13, 4, 15
STALL, MONOTONE = 0, 1
REGIME_NAMES = ("stall", "monotone")
KNEE = float(np.float32(0.8))
STALL_SLOPE = 1.5
THROTTLE_COST = 0.3
MATRIX_SHA256 = "289afb581b4fdbc97fd68566c0b42ff6244b20686e9e49ba177e7e6bd57454a1"
SOURCE_SHA256 = {
    "experiments/shift/thruster_chain.py": "f22f2e214ad9231bee4ce60ef55e64409534c2654c4e184406553ba8ed67dde1",
    "experiments/signal/gapped_lq.py": "f8ec7cc6c7d9d013d3e509cd3aba0a6dedf1f371480b4e1977f51e2b5dd923aa",
}

# Exact Python float repr of the genuine legacy matrix export, 2026-10-20.
# Stage0: radii .98-.96, angles .10/.20/.35/.55/.80, coupling norm .02,
# coupling seed 20261011, first accepted draw; no recalculation on import.
_A10 = np.array([
    [0.9777125812299784, -0.09672215307307072, -0.0032177474714957027, -0.0036854581926546795, 0.0022276001886515034, 5.0097360180674706e-05, 0.003607227365457634, -0.0025977450297937827, -0.0010500340801089907, 0.0009652492372484052],
    [0.09940391721844581, 0.9759393868366641, 0.0001307439011167588, -0.0014584202432089749, 0.004287444772601739, 0.006358675619747226, -0.003249106414894763, -0.0022826844542754962, -0.001195739582567211, 0.002141233362877725],
    [0.004727777116290652, 0.001676003589695229, 0.9551871912419472, -0.19400801385647914, 0.0066585559862546925, 0.004321981810486594, -0.002351504691480412, -0.0028796358576129743, -0.00017534637907377042, -0.0014908328910182977],
    [-0.0010694645348326218, 0.003950533344859924, 0.2002191805870674, 0.9498613915785109, -0.005257280339745227, 0.003893854725563954, -0.0016027860425690422, 0.002178774672635904, 0.0032342039892158244, 0.004745834820556732],
    [0.0005227925125596042, 0.0034722484793222414, 0.003327179871759413, -0.0011316590449246787, 0.9106172106519078, -0.33126273154497077, 0.0029036055232298766, -0.003299177904469415, -0.0002792636049624802, 0.0020210035590503706],
    [-0.004426956214867775, 0.0026020949999520306, -0.001087741967027858, 1.0090042320887824e-05, 0.33940832566241086, 0.9089986694657229, 0.002562491688699326, 0.00026117818132252545, -0.001954887543497725, -0.003806489583919625],
    [-0.002864014579333474, 0.005511555544683352, -0.0007795943457998527, 0.0010125392394960543, 0.00302278339678814, -0.0015069787253937821, 0.8217534382428903, -0.5054721930688784, 0.004390424840539214, -0.0027294911446361066],
    [0.0032829775189667934, 0.0017827723597421883, 0.0008687221328898804, -0.0002566114977840717, 0.005419950373129756, -0.0005569388845530562, 0.5049667331935627, 0.8227127339382514, 0.0033464391706002545, -0.00015333122670848354],
    [0.001872218383923559, 0.0106033926617112, 0.005927786293139656, -0.0012436923251405061, 0.0033156634642937137, -0.0013459488981572368, -0.0031035397212931673, 0.0018979088335239842, 0.6637979117655004, -0.6819570736534781],
    [-0.0019264007165151803, 0.004132407256634719, 0.0003150500063740507, -0.0038427711853132804, 0.008823080271840861, 0.0028660639483504276, 0.004346767798996067, -0.005234195917955739, 0.6890547118528255, 0.6655616509674585],
], dtype=np.float64)
_A = np.zeros((DX, DX), dtype=np.float64)
_A[:DS, :DS] = _A10
_A[1:DS:2, 10:12] = 0.3
_A[10, 10] = 1.0
_A[11:, 11:] = [[0.9637708963658905, -0.26673143668883115],
                  [0.26673143668883115, 0.9637708963658905]]
_B = np.zeros((DX, DA), dtype=np.float64)
for _k in range(DA):
    _B[2 * _k + 1, _k] = 0.3
    _B[2 * (_k + 1) + 1, _k] = 0.3
for _matrix in (_A10, _A, _B):
    _matrix.setflags(write=False)


def matrices():
    """Return the immutable shared plant matrices (13,13) and (13,4)."""
    return _A, _B


def provenance():
    """JSON-compatible plant provenance; source files are never executed here."""
    digest = hashlib.sha256(_A.astype("<f8").tobytes() + _B.astype("<f8").tobytes()).hexdigest()
    if digest != MATRIX_SHA256:
        raise RuntimeError("frozen ThrusterChain matrix digest mismatch")
    return dict(adapter="frozen-original-matrix-export-v1", matrix_sha256=digest,
                source_sha256=dict(SOURCE_SHA256), state_dim=DX, action_dim=DA,
                observation_dim=OBS_DIM, knee=KNEE, stall_slope=STALL_SLOPE,
                throttle_cost=THROTTLE_COST, arithmetic="float64 fixed-column order",
                original_default_gamma=0.98, diagnostic_default_gamma=1.0,
                terminal_bootstrap=False)


def regime_id(regime):
    if isinstance(regime, str):
        if regime == "no_stall":
            return MONOTONE
        if regime in REGIME_NAMES:
            return REGIME_NAMES.index(regime)
    elif regime in (STALL, MONOTONE):
        return int(regime)
    raise ValueError("regime must be 'stall'/0 or 'monotone'/1")


def _finite_vector(value, size, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape[-1:] != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with final dimension {size}")
    return result


def thrust(action, regime=STALL):
    """Original thrust arithmetic on an already clipped command, float64."""
    action = _finite_vector(action, DA, "action")
    knee, slope = (KNEE, STALL_SLOPE) if regime_id(regime) == STALL else (1.0, 0.0)
    return np.clip(action, -knee, knee) - slope * np.sign(action) * np.maximum(np.abs(action) - knee, 0.0)


def dynamics(state, action, regime=STALL):
    """Original deterministic step, including action clipping and reward order.

    Supports matching/broadcastable leading dimensions. No implicit float32
    boundary: that is provided separately by the environment and rollouts.
    """
    state = _finite_vector(state, DX, "state")
    action = np.clip(_finite_vector(action, DA, "action"), -1.0, 1.0)
    push = thrust(action, regime)
    nxt = state[..., 0:1] * _A[:, 0]
    for j in range(1, DX):
        nxt = nxt + state[..., j:j + 1] * _A[:, j]
    for k in range(DA):
        nxt = nxt + push[..., k:k + 1] * _B[:, k]
    s2 = state[..., 0] * state[..., 0]
    for j in range(1, DS):
        s2 = s2 + state[..., j] * state[..., j]
    a2 = action[..., 0] * action[..., 0]
    for k in range(1, DA):
        a2 = a2 + action[..., k] * action[..., k]
    return nxt, -(s2 + THROTTLE_COST * a2)


@dataclass(frozen=True)
class Objective:
    horizon: int = 100
    gamma: float = 1.0

    def __post_init__(self):
        if isinstance(self.horizon, bool) or int(self.horizon) != self.horizon or self.horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if not math.isfinite(self.gamma) or not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0,1]")
        object.__setattr__(self, "horizon", int(self.horizon))
        object.__setattr__(self, "gamma", float(self.gamma))

    def as_dict(self):
        return dict(horizon=self.horizon, gamma=self.gamma, terminal_reward=0.0,
                    terminal_bootstrap=False, reward="-(sum(state[:10]**2)+0.3*sum(action**2))")


PRIMARY_OBJECTIVE = Objective()


def _time(t, objective):
    if isinstance(t, bool) or int(t) != t or not 0 <= t <= objective.horizon:
        raise ValueError("time must be an integer between 0 and the horizon")
    return int(t)


def observation(state, regime, t, objective=PRIMARY_OBJECTIVE, *, float32=False):
    state = _finite_vector(state, DX, "state")
    t = _time(t, objective)
    context = np.broadcast_to([regime_id(regime), (objective.horizon - t) / objective.horizon],
                              state.shape[:-1] + (2,))
    return np.concatenate((state, context), axis=-1).astype(np.float32 if float32 else np.float64)


class ContinuousThrusterChain:
    """Dependency-light episode wrapper; regime is fixed by reset, never by step.

    Like the original Gym environment, horizon exhaustion is a truncation.
    Unlike continuing-task training, its bootstrap mask is zero there because
    the declared diagnostic objective ends at this exact horizon.
    """
    def __init__(self, objective=PRIMARY_OBJECTIVE, *, float32_deployment=False):
        self.objective = objective
        self.float32_deployment = bool(float32_deployment)
        self.state = None

    def reset(self, state, regime=STALL):
        state = _finite_vector(state, DX, "state")
        if state.ndim != 1:
            raise ValueError("reset requires one state")
        if (np.linalg.norm(state[:DS]) > 100 or not 0 <= state[10] <= 1.5
                or np.hypot(state[11], state[12]) > 1.0 + 1e-9):
            raise ValueError("reset state is outside the original ThrusterChain start set")
        self.state, self.regime, self.t = state.copy(), regime_id(regime), 0
        return observation(self.state, self.regime, 0, self.objective,
                           float32=self.float32_deployment), self._info(False)

    def _info(self, done):
        return dict(state=self.state.copy(), regime=self.regime, t=self.t,
                    bootstrap_mask=float(not done), finite_horizon_done=bool(done))

    def step(self, action):
        if self.state is None or self.t >= self.objective.horizon:
            raise RuntimeError("reset before stepping an absent or exhausted episode")
        action = _command(action, self.float32_deployment)
        self.state, reward = dynamics(self.state, action, self.regime)
        self.t += 1
        done = self.t == self.objective.horizon
        info = self._info(done)
        info["command"] = action.copy()
        return (observation(self.state, self.regime, self.t, self.objective,
                            float32=self.float32_deployment), float(reward), False, done, info)


def _command(action, float32_deployment=False):
    action = _finite_vector(action, DA, "action")
    if action.shape != (DA,):
        raise ValueError("one policy action must have shape (4,)")
    if float32_deployment:
        action = action.astype(np.float32).astype(np.float64)
    return np.clip(action, -1.0, 1.0)


@dataclass
class Rollout:
    states: np.ndarray             # (remaining+1,13)
    observations: np.ndarray       # (remaining+1,15)
    actions: np.ndarray            # (remaining,4), actual clipped commands
    rewards: np.ndarray            # (remaining,)
    return_value: float
    bootstrap_mask: np.ndarray     # (remaining,), final zero
    regime: int
    start_time: int
    objective: Objective
    float32_deployment: bool

    @property
    def cost(self):
        return -self.return_value


def rollout_policy(state, regime, policy: Callable, objective=PRIMARY_OBJECTIVE, *, t=0,
                   float32_deployment=False):
    """Exact deterministic rollout from a known state until the declared end.

    policy(obs15)->action4 must be deterministic for use as Q/V truth.  A
    stochastic collector may also use this routine, but must label its return
    as one sample, rather than the expected value of that stochastic policy.
    """
    t = _time(t, objective)
    regime = regime_id(regime)
    state = _finite_vector(state, DX, "state").copy()
    if state.shape != (DX,):
        raise ValueError("rollout requires one known state")
    remaining = objective.horizon - t
    states = np.empty((remaining + 1, DX))
    obs = np.empty((remaining + 1, OBS_DIM), dtype=np.float32 if float32_deployment else np.float64)
    actions, rewards = np.empty((remaining, DA)), np.empty(remaining)
    states[0] = state
    obs[0] = observation(state, regime, t, objective, float32=float32_deployment)
    for k in range(remaining):
        actions[k] = _command(policy(obs[k].copy()), float32_deployment)
        state, rewards[k] = dynamics(state, actions[k], regime)
        states[k + 1] = state
        obs[k + 1] = observation(state, regime, t + k + 1, objective, float32=float32_deployment)
    mask = np.ones(remaining, dtype=np.float64)
    if remaining:
        mask[-1] = 0.0
    value = float(np.dot(objective.gamma ** np.arange(remaining), rewards))
    return Rollout(states, obs, actions, rewards, value, mask, regime, t, objective, bool(float32_deployment))


def rollout_plan(state, regime, actions, objective=PRIMARY_OBJECTIVE, *, t=0, float32_deployment=False):
    t = _time(t, objective)
    actions = np.asarray(actions, dtype=np.float64)
    if actions.shape != (objective.horizon - t, DA) or not np.all(np.isfinite(actions)):
        raise ValueError("plan must be finite with shape (horizon-t,4)")
    iterator = iter(actions)
    return rollout_policy(state, regime, lambda _: next(iterator), objective, t=t,
                          float32_deployment=float32_deployment)


def policy_value(state, regime, policy, objective=PRIMARY_OBJECTIVE, *, t=0, float32_deployment=False):
    """Finite V_t^pi from a known state, deterministic rollout with local discount."""
    return rollout_policy(state, regime, policy, objective, t=t,
                          float32_deployment=float32_deployment).return_value


def policy_q(state, regime, action, policy, objective=PRIMARY_OBJECTIVE, *, t=0, float32_deployment=False):
    """Finite Q_t^pi: one forced action and deterministic policy thereafter.

    Q is undefined at the terminal observation; V_H is exactly zero.
    """
    t = _time(t, objective)
    if t == objective.horizon:
        raise ValueError("Q has no action at the terminal observation")
    nxt, reward = dynamics(state, _command(action, float32_deployment), regime)
    return float(reward) + objective.gamma * policy_value(
        nxt, regime, policy, objective, t=t + 1, float32_deployment=float32_deployment)


def dominated_stall_action(action):
    """Equivalent low-throttle command, with no greater energy cost.

    For clipped |a|>k, slope=1.5 and k=float32(.8), |thrust(a)|<=k<|a|.
    Replacing a by thrust(a) preserves the entire successor state and strictly
    reduces its energy cost. Thus every full-range stalled plan is dominated
    by a plan in [-k,k]^4, including for every positive discount weight.
    """
    return thrust(np.clip(_finite_vector(action, DA, "action"), -1.0, 1.0), STALL)


@dataclass
class QPSolution:
    actions: np.ndarray
    cost: float
    return_value: float
    fw_gap: float
    relative_fw_gap: float
    replay_error: float
    status: int
    iterations: int
    cap: float
    regime: int
    objective: Objective
    start_time: int
    solver: str

    @property
    def return_interval(self):
        """Numerical convex certificate (not an outward-rounded interval)."""
        return (self.return_value, self.return_value + self.fw_gap)

    def as_dict(self):
        return dict(cost=self.cost, return_value=self.return_value, fw_gap=self.fw_gap,
                    relative_fw_gap=self.relative_fw_gap, replay_error=self.replay_error,
                    status=self.status, iterations=self.iterations, cap=self.cap,
                    regime=REGIME_NAMES[self.regime], objective=self.objective.as_dict(),
                    start_time=self.start_time, solver=self.solver,
                    return_interval=list(self.return_interval),
                    certificate="numerical Frank-Wolfe gap; floating-point, not interval arithmetic")


class FiniteHorizonQP:
    """Strictly convex bounded least squares for the finite-horizon optimum.

    M u + offset(x0) stacks sqrt(gamma**k)*s_k and
    sqrt(.3*gamma**k)*u_k for k=0..remaining-1. No x_H cost occurs.
    In stall the dominated-command theorem permits cap=KNEE; monotone cap=1.
    A smaller explicit cap evaluates the optimum under that additional limit.
    """
    def __init__(self, objective=PRIMARY_OBJECTIVE, *, t=0):
        self.objective, self.start_time = objective, _time(t, objective)
        self.length = objective.horizon - self.start_time
        self.M, self.free_response, self.Q, self.R = _least_squares(self.length, objective.gamma)

    def offset(self, state):
        state = _finite_vector(state, DX, "state")
        if state.shape != (DX,):
            raise ValueError("QP requires one known state")
        return np.concatenate(((self.free_response @ state).reshape(-1), np.zeros(self.length * DA)))

    def cost(self, actions, state):
        action = np.asarray(actions, dtype=np.float64)
        if action.size != self.length * DA or not np.all(np.isfinite(action)):
            raise ValueError("actions have wrong size or contain nonfinite values")
        residual = self.M @ action.reshape(-1) + self.offset(state)
        return float(residual @ residual)

    def gradient(self, actions, state):
        flat = np.asarray(actions, dtype=np.float64).reshape(-1)
        return 2.0 * self.M.T @ (self.M @ flat + self.offset(state))

    def frank_wolfe_gap(self, actions, state, cap):
        flat = np.asarray(actions, dtype=np.float64).reshape(-1)
        if np.any(np.abs(flat) > cap + 1e-12):
            raise ValueError("Frank-Wolfe certificate requires a feasible plan")
        gradient = self.gradient(flat, state)
        return float(max(0.0, np.sum(gradient * flat + cap * np.abs(gradient))))

    def solve(self, state, regime=STALL, *, cap=None, method="bvls", tol=1e-12, max_iter=20000):
        regime = regime_id(regime)
        physical_cap = KNEE if regime == STALL else 1.0
        cap = physical_cap if cap is None else float(cap)
        if not math.isfinite(cap) or not 0 <= cap <= physical_cap:
            raise ValueError("QP cap must lie in [0,KNEE] for stall or [0,1] for monotone")
        if method not in ("bvls", "trf"):
            raise ValueError("method must be 'bvls' or 'trf'")
        offset = self.offset(state)
        if self.length == 0 or cap == 0:
            flat, status, iterations = np.zeros(self.length * DA), 3, 0
        else:
            result = lsq_linear(self.R, -(self.Q.T @ offset), bounds=(-cap, cap), method=method,
                                tol=tol, max_iter=max_iter, lsq_solver="exact")
            if not result.success:
                raise RuntimeError(f"bounded least squares did not converge: {result.message}")
            flat, status, iterations = np.clip(result.x, -cap, cap), int(result.status), int(result.nit)
        actions = flat.reshape(self.length, DA)
        cost = self.cost(actions, state)
        gap = self.frank_wolfe_gap(actions, state, cap)
        replay = rollout_plan(state, regime, actions, self.objective, t=self.start_time)
        return QPSolution(actions, cost, -cost, gap, gap / max(1.0, cost), abs(cost - replay.cost),
                          status, iterations, cap, regime, self.objective, self.start_time, method)


@lru_cache(maxsize=16)
def _least_squares(length, gamma):
    powers = [np.eye(DX)]
    for _ in range(length):
        powers.append(_A @ powers[-1])
    free = np.zeros((length, DS, DX))
    state_map = np.zeros((length * DS, length * DA))
    weights = gamma ** (0.5 * np.arange(length))
    if length and np.any(weights * weights == 0):
        raise ValueError("discount/horizon underflows a cost weight; use a shorter horizon")
    for t in range(length):
        free[t] = weights[t] * powers[t][:DS]
        for k in range(t):
            state_map[t * DS:(t + 1) * DS, k * DA:(k + 1) * DA] = (
                weights[t] * (powers[t - 1 - k] @ _B)[:DS])
    energy = np.diag(np.repeat(math.sqrt(THROTTLE_COST) * weights, DA))
    M = np.vstack((state_map, energy))
    Q, R = qr(M, mode="economic") if length else (np.zeros((0, 0)), np.zeros((0, 0)))
    for array in (M, free, Q, R):
        array.setflags(write=False)
    return M, free, Q, R


@lru_cache(maxsize=16)
def _lqr_gains(objective):
    """Unconstrained finite-horizon linear gains, clipped only by named controls."""
    Q = np.diag([1.0] * DS + [0.0] * (DX - DS))
    R, P = THROTTLE_COST * np.eye(DA), np.zeros((DX, DX))
    gains = np.zeros((objective.horizon + 1, DA, DX))
    for t in range(objective.horizon - 1, -1, -1):
        gains[t] = np.linalg.solve(R + objective.gamma * _B.T @ P @ _B,
                                   objective.gamma * _B.T @ P @ _A)
        P = Q + objective.gamma * _A.T @ P @ (_A - _B @ gains[t])
        P = 0.5 * (P + P.T)
    gains.setflags(write=False)
    return gains


class CappedLQRPolicy:
    """Finite-horizon LQR followed by a declared cap; not the constrained oracle."""
    def __init__(self, objective=PRIMARY_OBJECTIVE, *, cap=None):
        if cap is not None and (not math.isfinite(cap) or not 0 <= cap <= 1):
            raise ValueError("fixed cap must lie in [0,1]")
        self.objective, self.cap = objective, cap
        self.gains = _lqr_gains(objective)
        self.name = "regime_cap" if cap is None else f"fixed_cap_{cap:g}"

    def __call__(self, obs):
        obs = _finite_vector(obs, OBS_DIM, "observation")
        if obs.shape != (OBS_DIM,):
            raise ValueError("controller requires one observation")
        t = int(round(self.objective.horizon * (1.0 - float(obs[-1]))))
        _time(t, self.objective)
        regime = regime_id(obs[-2])
        cap = (KNEE if regime == STALL else 1.0) if self.cap is None else self.cap
        return np.clip(-self.gains[t] @ obs[:DX], -cap, cap)


def fixed_cap_policy(cap=KNEE, objective=PRIMARY_OBJECTIVE):
    return CappedLQRPolicy(objective, cap=float(cap))


def regime_cap_policy(objective=PRIMARY_OBJECTIVE):
    return CappedLQRPolicy(objective, cap=None)


def high_load_start():
    """Named deterministic witness in the original start set; not a tuned bank."""
    state = np.zeros(DX)
    state[10] = 1.5
    return state


def verify_legacy_parity(legacy, *, repository_root=None, seed=20261020, n=32):
    """Check an original module imported normally; never fake its dependencies.

    If repository_root is given, also require raw-byte hashes of the two
    authoritative source files to match the recorded export provenance.
    The input legacy module must expose the original matrices/dynamics/thrust.
    """
    provenance()
    hashes = {}
    if repository_root is not None:
        root = Path(repository_root)
        for rel, expected in SOURCE_SHA256.items():
            actual = hashlib.sha256((root / rel).read_bytes()).hexdigest()
            if actual != expected:
                raise AssertionError(f"legacy source changed: {rel}")
            hashes[rel] = actual
    for frozen, original in zip(matrices(), legacy.matrices(legacy.STALL)):
        np.testing.assert_array_equal(frozen, original)
    rng = np.random.default_rng(seed)
    states = rng.normal(size=(n, DX))
    actions = rng.uniform(-1.2, 1.2, size=(n, DA))
    special = np.array([[KNEE, -KNEE, np.nextafter(KNEE, np.inf), -1.0],
                        [1.0, 0.0, np.float32(0.8), -0.9]])
    actions[:min(n, 2)] = special[:min(n, 2)]
    for regime, physics in ((STALL, legacy.STALL), (MONOTONE, legacy.NO_STALL)):
        np.testing.assert_array_equal(thrust(actions, regime), legacy.thrust(actions, physics))
        ours, reward = dynamics(states, actions, regime)
        original, original_reward = legacy.dynamics(states, actions, physics)
        np.testing.assert_array_equal(ours, original)
        np.testing.assert_array_equal(reward, original_reward)
        for i in range(n):
            one, r = dynamics(states[i], actions[i], regime)
            np.testing.assert_array_equal(ours[i], one)
            np.testing.assert_equal(reward[i], r)
    return dict(passed=True, equality="bitwise float64", regimes=list(REGIME_NAMES),
                random_cases_per_regime=n, source_sha256=hashes, matrix_sha256=MATRIX_SHA256)

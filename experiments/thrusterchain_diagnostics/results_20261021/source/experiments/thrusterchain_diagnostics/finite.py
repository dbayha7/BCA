"""Small, exactly solvable ThrusterChain-inspired finite MDP.

This is a *companion model*, not an exact abstraction of the 13-dimensional
continuous environment. The explicit kernel is part of its definition: damped
position plus load plus thrust is clipped to the position bounds, then rounded
randomly to the two adjacent grid points with linear interpolation probabilities.
This preserves the clipped position's conditional mean. Regime and load are
observed and episode-fixed. Time is external to the physical state and every
policy/value/occupancy has a separate finite-horizon time axis.

The five-action grid deliberately omits the approximately .5 low command with
the same thrust as a stalled full command. Consequently the continuous model's
universal action-replacement dominance proof does not apply to this companion:
some small-displacement stall states can prefer full command to the available
grid alternatives. The intended harmful-versus-useful contrast is the declared
positive-displacement/high-load critical region, with a tested Q witness at
position 1.5 and time 0; it is not a global finite-action dominance claim.

Normal observations are (position, regime, load); regime 0 stalls and regime 1
is monotone. Hidden-world controls remove regime entirely from observations and
the state index. Their two kernels agree on every common-support action. World
identity belongs to the evaluator, never to learner observations or data.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


KNEE = float(np.float32(0.8))
REGIME_NAMES = ("stall", "monotone")
CONTEXT_MIXES = ("overlap", "thin", "zero")


@dataclass(frozen=True)
class FiniteConfig:
    horizon: int = 8
    n_positions: int = 9
    gamma: float = 0.98
    position_bound: float = 2.0
    damping: float = 0.9
    gain: float = 0.65
    loads: tuple = (0.1, 0.35)
    throttle_cost: float = 0.06
    stall_slope: float = 1.5

    def __post_init__(self):
        if not isinstance(self.horizon, (int, np.integer)) or self.horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if not isinstance(self.n_positions, (int, np.integer)) or self.n_positions < 3:
            raise ValueError("n_positions must be an integer >= 3")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        if not np.isfinite(self.position_bound) or self.position_bound <= 0:
            raise ValueError("position_bound must be positive and finite")
        if len(self.loads) != 2 or not np.all(np.isfinite(self.loads)):
            raise ValueError("loads must contain exactly two finite values")
        if not np.all(np.isfinite([self.damping, self.gain, self.throttle_cost, self.stall_slope])):
            raise ValueError("physical parameters must be finite")
        if self.gain <= 0 or self.throttle_cost < 0 or self.stall_slope < 0:
            raise ValueError("gain must be positive and cost/slope nonnegative")


@dataclass(frozen=True)
class DPResult:
    V: np.ndarray             # [H+1, S], zero terminal value
    Q: np.ndarray             # [H, S, A]
    policy: np.ndarray        # [H, S, A]

    def start_value(self, initial):
        return float(np.asarray(initial) @ self.V[0])


@dataclass(frozen=True)
class OccupancyResult:
    state: np.ndarray         # [H+1, S], normalized at each time
    state_action: np.ndarray  # [H, S, A], normalized at each time


@dataclass(frozen=True)
class RatioResult:
    """Exact ratios; unsupported positive target / zero source is NaN.

    supported means source > 0. Both-zero cells also have NaN, but are not an
    overlap violation. unsupported is exactly (target > 0) & (source == 0).
    No epsilon, clipping, or implicit support extrapolation is used.
    """
    ratio: np.ndarray
    supported: np.ndarray
    unsupported: np.ndarray
    target_mass_unsupported: float


class FiniteMDP:
    def __init__(self, config: FiniteConfig = FiniteConfig(), hidden_world: Optional[str] = None):
        if hidden_world == "no_stall":
            hidden_world = "monotone"
        if hidden_world not in (None, "stall", "monotone"):
            raise ValueError("hidden_world must be None, 'stall', or 'monotone'")
        self.config = config
        self.hidden_world = hidden_world
        self.horizon = config.horizon
        self.gamma = config.gamma
        self.positions = np.linspace(-config.position_bound, config.position_bound, config.n_positions)
        self.actions = np.array([-1.0, -KNEE, 0.0, KNEE, 1.0])
        self.n_actions = len(self.actions)
        self.n_regimes = 2 if hidden_world is None else 1
        self.n_loads = len(config.loads)
        self.n_states = self.n_regimes * self.n_loads * config.n_positions
        regime, load, pos = np.indices((self.n_regimes, self.n_loads, config.n_positions))
        self.state_regime = regime.ravel()
        self.state_load = load.ravel()
        self.position_index = pos.ravel()
        self.state_position = self.positions[self.position_index]
        columns = [self.state_position, self.state_regime, self.state_load]
        if hidden_world is not None:
            columns.pop(1)
        self.observations = np.column_stack(columns).astype(np.float64)
        self.common_action_mask = np.abs(self.actions) <= KNEE
        self.P, self.R = self._build_kernel()
        for value in (self.positions, self.actions, self.state_regime, self.state_load,
                      self.position_index, self.state_position, self.observations,
                      self.common_action_mask, self.P, self.R):
            value.setflags(write=False)

    def state_index(self, position_index, load=0, regime=0):
        """Index by *grid index*, not by physical position."""
        if not all(isinstance(value, (int, np.integer)) for value in (position_index, load, regime)):
            raise ValueError("state coordinates must be integer indices")
        if not (0 <= position_index < self.config.n_positions and 0 <= load < self.n_loads
                and 0 <= regime < self.n_regimes):
            raise ValueError("state coordinate outside finite grid")
        return int((regime * self.n_loads + load) * self.config.n_positions + position_index)

    def thrust(self, action, regime=0):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        stall = (self.hidden_world == "stall" if self.hidden_world is not None
                 else np.asarray(regime) == 0)
        stalled = np.clip(action, -KNEE, KNEE) - self.config.stall_slope * np.sign(action) * np.maximum(np.abs(action) - KNEE, 0)
        return np.where(stall, stalled, action)

    def quantize(self, position):
        """Grid-mass vector for one bounded position; boundary clips explicitly."""
        if not np.isfinite(position):
            raise ValueError("position must be finite")
        position = float(np.clip(position, self.positions[0], self.positions[-1]))
        upper = int(np.searchsorted(self.positions, position, side="right"))
        mass = np.zeros(self.config.n_positions)
        if upper == 0:
            mass[0] = 1.0
        elif upper == self.config.n_positions:
            mass[-1] = 1.0
        else:
            lower = upper - 1
            weight = (position - self.positions[lower]) / (self.positions[upper] - self.positions[lower])
            mass[lower], mass[upper] = 1.0 - weight, weight
        return mass

    def _build_kernel(self):
        P = np.zeros((self.n_states, self.n_actions, self.n_states))
        R = -self.state_position[:, None] ** 2 - self.config.throttle_cost * self.actions[None, :] ** 2
        for state in range(self.n_states):
            regime, load = self.state_regime[state], self.state_load[state]
            block_start = self.state_index(0, load, regime)
            for action in range(self.n_actions):
                next_position = (self.config.damping * self.state_position[state]
                                 + self.config.gain * self.thrust(self.actions[action], regime)
                                 + self.config.loads[load])
                P[state, action, block_start:block_start + self.config.n_positions] = self.quantize(next_position)
        return P, R

    def critical_mask(self):
        """Pre-outcome subset: positive displacement, high load, negative high action.

        This uses only the current state and intended action. Neither reward,
        next state, future occupancy, optimal Q, nor world identity is consulted.
        It defines an experimental subset, not a guarantee that every selected
        time/state/action has the same sign of optimal-Q contrast.
        """
        return ((self.state_position[:, None] >= 0.5)
                & (self.state_load[:, None] == 1)
                & (self.actions[None, :] < -KNEE))

    def initial_distribution(self, context_mix="overlap"):
        """Start law independent of retention and policy.

        overlap: uniform regime/load contexts; thin: 2% high-load mass; zero:
        no high-load mass. Within each context positions nearest .5, 1, 1.5
        have equal mass. An explicit [n_regimes, 2] probability table is allowed.
        """
        if isinstance(context_mix, str):
            if context_mix not in CONTEXT_MIXES:
                raise ValueError("context_mix must be 'overlap', 'thin', or 'zero'")
            loads = {"overlap": (0.5, 0.5), "thin": (0.98, 0.02), "zero": (1.0, 0.0)}[context_mix]
            contexts = np.tile(loads, (self.n_regimes, 1)) / self.n_regimes
        else:
            contexts = np.asarray(context_mix, dtype=float)
            if contexts.shape != (self.n_regimes, self.n_loads):
                raise ValueError("context weights must have shape [n_regimes, n_loads]")
            _check_probability_array(contexts, "context weights")
            if not np.isclose(contexts.sum(), 1.0, rtol=0, atol=1e-12):
                raise ValueError("context weights must sum to one")
        selected = np.unique([np.argmin(np.abs(self.positions - x)) for x in (0.5, 1.0, 1.5)])
        initial = np.zeros(self.n_states)
        for regime in range(self.n_regimes):
            for load in range(self.n_loads):
                for pos in selected:
                    initial[self.state_index(pos, load, regime)] = contexts[regime, load] / len(selected)
        return initial


def _check_probability_array(values, name):
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"{name} must be finite and nonnegative")


def as_policy(mdp, policy):
    """Validate probabilities; broadcast stationary [S,A] or constant [A]."""
    policy = np.asarray(policy, dtype=float)
    shape = (mdp.horizon, mdp.n_states, mdp.n_actions)
    if policy.shape not in ((mdp.n_actions,), (mdp.n_states, mdp.n_actions), shape):
        raise ValueError("policy must have shape [A], [S,A], or [H,S,A]")
    policy = np.broadcast_to(policy, shape)
    _check_probability_array(policy, "policy")
    if not np.allclose(policy.sum(axis=-1), 1.0, rtol=0, atol=1e-12):
        raise ValueError("each policy probability row must sum to one")
    return policy


def solve_optimal(mdp, action_mask=None):
    """Exact finite-horizon backward induction; deterministic first-index tie rule.

    Optional action_mask [A], [S,A], or [H,S,A] limits the optimizing policy.
    Q still reports every one-step action followed by the constrained policy;
    forbidden-action Q entries are not replaced with arbitrary sentinel values.
    """
    if action_mask is None:
        action_mask = np.ones((mdp.horizon, mdp.n_states, mdp.n_actions), dtype=bool)
    else:
        action_mask = np.asarray(action_mask)
        if action_mask.dtype != bool:
            raise ValueError("action_mask must contain booleans")
        if action_mask.shape not in ((mdp.n_actions,), (mdp.n_states, mdp.n_actions),
                                     (mdp.horizon, mdp.n_states, mdp.n_actions)):
            raise ValueError("action_mask must have shape [A], [S,A], or [H,S,A]")
        action_mask = np.broadcast_to(action_mask, (mdp.horizon, mdp.n_states, mdp.n_actions))
        if not action_mask.any(axis=-1).all():
            raise ValueError("each action_mask row must allow at least one action")
    V = np.zeros((mdp.horizon + 1, mdp.n_states))
    Q = np.empty((mdp.horizon, mdp.n_states, mdp.n_actions))
    policy = np.zeros_like(Q)
    for t in range(mdp.horizon - 1, -1, -1):
        Q[t] = mdp.R + mdp.gamma * np.einsum("sak,k->sa", mdp.P, V[t + 1])
        chosen = np.argmax(np.where(action_mask[t], Q[t], -np.inf), axis=-1)
        V[t] = Q[t, np.arange(mdp.n_states), chosen]
        policy[t, np.arange(mdp.n_states), chosen] = 1.0
    return DPResult(V, Q, policy)


def cap_policy(mdp, policy, cap=KNEE, regime_aware=False):
    """Push a common raw policy through an action cap, preserving exact masses.

    The cap must be a nonnegative grid action magnitude. regime_aware applies
    it only to the observed stall regime. This is a matched intervention on
    one raw policy, distinct from optimizing separate policies under masks.
    """
    policy = as_policy(mdp, policy)
    if not np.any(np.isclose(np.abs(mdp.actions), cap, rtol=0, atol=1e-12)) or cap < 0:
        raise ValueError("cap must be a nonnegative magnitude in the action grid")
    if regime_aware and mdp.hidden_world is not None:
        raise ValueError("regime-aware cap requires an observed regime")
    capped = np.zeros_like(policy)
    for s in range(mdp.n_states):
        active = not regime_aware or mdp.state_regime[s] == 0
        for a, command in enumerate(mdp.actions):
            clipped = np.clip(command, -cap, cap) if active else command
            target = int(np.argmin(abs(mdp.actions - clipped)))
            capped[:, s, target] += policy[:, s, a]
    return capped


def evaluate_policy(mdp, policy):
    """Q takes one action then follows the supplied policy at later times."""
    policy = as_policy(mdp, policy)
    V = np.zeros((mdp.horizon + 1, mdp.n_states))
    Q = np.empty((mdp.horizon, mdp.n_states, mdp.n_actions))
    for t in range(mdp.horizon - 1, -1, -1):
        Q[t] = mdp.R + mdp.gamma * np.einsum("sak,k->sa", mdp.P, V[t + 1])
        V[t] = np.sum(policy[t] * Q[t], axis=-1)
    return DPResult(V, Q, policy.copy())


def occupancies(mdp, policy, initial=None):
    """Undiscounted time-indexed occupancy under the *actual* supplied policy."""
    policy = as_policy(mdp, policy)
    if initial is None:
        initial = mdp.initial_distribution()
    initial = np.asarray(initial, dtype=float)
    _check_probability_array(initial, "initial distribution")
    if initial.shape != (mdp.n_states,) or not np.isclose(initial.sum(), 1.0, rtol=0, atol=1e-12):
        raise ValueError("initial must have shape [S] and sum to one")
    state = np.zeros((mdp.horizon + 1, mdp.n_states))
    state_action = np.zeros((mdp.horizon, mdp.n_states, mdp.n_actions))
    state[0] = initial
    for t in range(mdp.horizon):
        state_action[t] = state[t, :, None] * policy[t]
        state[t + 1] = np.einsum("sa,sak->k", state_action[t], mdp.P)
    return OccupancyResult(state, state_action)


def support_ratio(target, source):
    """Probability/occupancy ratio retaining genuine zero support as unsupported."""
    target, source = np.broadcast_arrays(np.asarray(target, float), np.asarray(source, float))
    _check_probability_array(target, "target mass")
    _check_probability_array(source, "source mass")
    supported = source > 0
    unsupported = (target > 0) & ~supported
    ratio = np.full(target.shape, np.nan)
    np.divide(target, source, out=ratio, where=supported)
    return RatioResult(ratio, supported, unsupported, float(target[unsupported].sum()))


def occupancy_ratio(mdp, target_policy, source_policy, target_initial=None, source_initial=None):
    target = occupancies(mdp, target_policy, target_initial).state_action
    source = occupancies(mdp, source_policy, source_initial).state_action
    return support_ratio(target, source)


def action_likelihood_ratio(mdp, target_policy, behavior_policy):
    return support_ratio(as_policy(mdp, target_policy), as_policy(mdp, behavior_policy))


def reachable_states(mdp, policy, initial=None):
    """Exact positive-support reachability; no numerical epsilon threshold."""
    return occupancies(mdp, policy, initial).state > 0


def hidden_kernel_certificate(config=FiniteConfig()):
    """Certify equality on every common action, hence on all reachable log states.

    Full finite-kernel equality on the action support is stronger than checking
    sampled logs. It proves all finite trajectory laws coincide for any shared
    start law and any logging policy supported on those actions. Equal rewards
    and observations also make the complete learner logs observationally equal.
    """
    stall = FiniteMDP(config, hidden_world="stall")
    monotone = FiniteMDP(config, hidden_world="monotone")
    common = stall.common_action_mask
    kernels_equal = np.array_equal(stall.P[:, common], monotone.P[:, common])
    rewards_equal = np.array_equal(stall.R[:, common], monotone.R[:, common])
    observations_equal = np.array_equal(stall.observations, monotone.observations)
    initial_equal = np.array_equal(stall.initial_distribution(), monotone.initial_distribution())
    probe_kernel_diff = np.max(np.abs(stall.P[:, ~common] - monotone.P[:, ~common]))
    return dict(kernels_equal=bool(kernels_equal), rewards_equal=bool(rewards_equal),
                observations_equal=bool(observations_equal), initial_equal=bool(initial_equal),
                all_common_state_action_pairs=int(stall.n_states * common.sum()),
                probe_kernel_max_difference=float(probe_kernel_diff),
                trajectory_laws_equal=bool(kernels_equal and rewards_equal and observations_equal and initial_equal))

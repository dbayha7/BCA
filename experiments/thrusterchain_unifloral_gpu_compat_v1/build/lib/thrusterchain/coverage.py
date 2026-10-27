"""Exact *designed-query* laws, never a closed-loop occupancy model.

The nine broad/rare/absent cells are independent terminal diagnostic query
distributions. They are not replacements for the benchmark trajectory datasets.
All densities refer to float64 physical states and continuous requested commands
with respect to 13D and 4D Lebesgue measure. The response can include float32
execution, but these densities are not masses of the quantized executed actions.
"""
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
import math
from typing import Mapping

import numpy as np


COVERAGE_LEVELS = ("broad", "rare", "absent")
STATE_LOW = np.array([-0.5] * 10 + [0.1, -0.2, -0.2])
STATE_HIGH = np.array([0.5] * 10 + [1.1, 0.2, 0.2])
STATE_SPLIT = 0.0
ACTION_SPLIT = 0.5
STATE_GAP_MASS = 0.5
ACTION_CORE_VOLUME = 1.5 * 2.0**3
ACTION_VOLUME = 2.0 ** 4
ACTION_GAP_VOLUME = ACTION_VOLUME - ACTION_CORE_VOLUME
ACTION_GAP_MASS = ACTION_GAP_VOLUME / ACTION_VOLUME
STATE_VOLUME = float(np.prod(STATE_HIGH - STATE_LOW))
STATE_LOW.setflags(write=False)
STATE_HIGH.setflags(write=False)


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _arrays(states, actions):
    states, actions = np.asarray(states), np.asarray(actions)
    if states.dtype.kind not in "fiu" or actions.dtype.kind not in "fiu":
        raise ValueError("queries must be real numeric arrays")
    states, actions = states.astype(float), actions.astype(float)
    if states.ndim != 2 or states.shape[1] != 13 or actions.shape != (len(states), 4):
        raise ValueError("states and actions must have shapes (n, 13) and (n, 4)")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("queries must be finite")
    return states, actions


def _gap_mass(level, broad_mass, rare_mass):
    return {"broad": broad_mass, "rare": rare_mass, "absent": 0.0}[level]


def _group_ids(group_ids, *, allow_empty=False):
    groups = np.asarray(group_ids)
    if groups.ndim != 1 or (not allow_empty and len(groups) == 0) or groups.dtype.kind not in "SUfiu":
        raise ValueError("group_ids must be a one-dimensional vector of strings or finite numbers")
    if groups.dtype.kind in "fiu" and not np.isfinite(groups).all():
        raise ValueError("numeric group_ids must be finite")
    return groups


@dataclass(frozen=True)
class ControlledQueries:
    states: np.ndarray
    actions: np.ndarray
    group_ids: np.ndarray
    state_gap: np.ndarray
    action_gap: np.ndarray
    law_name: str


@dataclass(frozen=True)
class CoverageLaw:
    """Product law with broad, rare, or absent mass in each declared gap.

    State gap C: first physical coordinate > 0 (normalized z1 > 0).
    Action gap B: a0 > .5. Broad is uniform over the whole containing box;
    rare allocates state_rare_mass=.05 or action_rare_mass=.01 to its gap; absent
    allocates zero. Conditional draws are uniform in each region.
    """
    state_coverage: str = "broad"
    action_coverage: str = "broad"
    state_rare_mass: float = 0.05
    action_rare_mass: float = 0.01

    def __post_init__(self):
        if self.state_coverage not in COVERAGE_LEVELS or self.action_coverage not in COVERAGE_LEVELS:
            raise ValueError("coverage must be broad, rare, or absent")
        for value, broad in ((self.state_rare_mass, STATE_GAP_MASS), (self.action_rare_mass, ACTION_GAP_MASS)):
            if (not isinstance(value, (int, float, np.integer, np.floating))
                    or isinstance(value, (bool, np.bool_)) or not np.isfinite(value)
                    or not np.sqrt(np.finfo(float).tiny) <= value < broad):
                raise ValueError("each rare mass must be a real scalar >= sqrt(float64.tiny) and smaller than its broad gap mass")

    @property
    def name(self):
        return f"state_{self.state_coverage}__action_{self.action_coverage}"

    @property
    def state_gap_mass(self):
        return _gap_mass(self.state_coverage, STATE_GAP_MASS, self.state_rare_mass)

    @property
    def action_gap_mass(self):
        return _gap_mass(self.action_coverage, ACTION_GAP_MASS, self.action_rare_mass)

    @property
    def region_masses(self):
        s, a = self.state_gap_mass, self.action_gap_mass
        return {"core_core": (1-s)*(1-a), "gap_core": s*(1-a),
                "core_gap": (1-s)*a, "gap_gap": s*a}

    def sample(self, n, rng=None, *, group_prefix="query"):
        n = _integer(n, "n")
        rng = np.random.default_rng(rng)
        state_gap = rng.random(n) < self.state_gap_mass
        action_gap = rng.random(n) < self.action_gap_mass
        states = rng.uniform(STATE_LOW, STATE_HIGH, size=(n, 13))
        states[~state_gap, 0] = rng.uniform(STATE_LOW[0], STATE_SPLIT, np.sum(~state_gap))
        states[state_gap, 0] = rng.uniform(np.nextafter(STATE_SPLIT, np.inf), STATE_HIGH[0], np.sum(state_gap))
        actions = rng.uniform(-1., 1., size=(n, 4))
        actions[~action_gap, 0] = rng.uniform(-1., ACTION_SPLIT, np.sum(~action_gap))
        actions[action_gap, 0] = rng.uniform(np.nextafter(ACTION_SPLIT, np.inf), 1., np.sum(action_gap))
        groups = np.array([f"{group_prefix}:{i}" for i in range(n)], dtype=str)
        return ControlledQueries(states, actions, groups, state_gap, action_gap, self.name)

    def density(self, states, actions):
        states, actions = _arrays(states, actions)
        state_inside = np.all((states >= STATE_LOW) & (states <= STATE_HIGH), axis=1)
        action_inside = np.all(np.abs(actions) <= 1.0, axis=1)
        sg, ag = states[:, 0] > STATE_SPLIT, actions[:, 0] > ACTION_SPLIT
        ps, pa = self.state_gap_mass, self.action_gap_mass
        state_pdf = np.where(sg, ps/(STATE_VOLUME*STATE_GAP_MASS),
                             (1-ps)/(STATE_VOLUME*(1-STATE_GAP_MASS)))
        action_pdf = np.where(ag, pa/ACTION_GAP_VOLUME, (1-pa)/ACTION_CORE_VOLUME)
        return np.where(state_inside & action_inside, state_pdf*action_pdf, 0.0)

    def logpdf(self, states, actions):
        with np.errstate(divide="ignore"):
            return np.log(self.density(states, actions))

    def density_ratio(self, states, actions, target=None):
        """Target/source ratio: +inf for positive target mass outside support.

        Zero over zero is represented as zero, outside both distributions.
        Never clip, cap, or renormalize these exact ratios.
        """
        target = target or CoverageLaw()
        source_pdf, target_pdf = self.density(states, actions), target.density(states, actions)
        ratio = np.zeros_like(source_pdf)
        np.divide(target_pdf, source_pdf, out=ratio, where=source_pdf > 0)
        ratio[(target_pdf > 0) & (source_pdf == 0)] = np.inf
        return ratio

    def support_report(self, target=None):
        target = target or CoverageLaw()
        missing_state = target.state_gap_mass if self.state_gap_mass == 0 else 0.0
        missing_action = target.action_gap_mass if self.action_gap_mass == 0 else 0.0
        unsupported = missing_state + missing_action - missing_state*missing_action
        supported = missing_state == 0 and missing_action == 0
        return {"status": "supported" if supported else "unsupported",
                "absolute_continuity": supported,
                "target_unsupported_mass": float(unsupported),
                "missing_state_mass": float(missing_state), "missing_action_mass": float(missing_action),
                "accounting": "exact analytic designed-query density", "occupancy_claim": False}

    def ratio_report(self, states, actions, target=None):
        """Flag outside-domain 0/0 separately from positive-target missing support."""
        target = target or CoverageLaw()
        source_pdf, target_pdf = self.density(states, actions), target.density(states, actions)
        status = np.full(len(source_pdf), "supported_target", dtype="<U24")
        status[(source_pdf == 0) & (target_pdf == 0)] = "outside_both_laws"
        status[(source_pdf > 0) & (target_pdf == 0)] = "source_only"
        status[(source_pdf == 0) & (target_pdf > 0)] = "unsupported_target"
        return {"ratio": self.density_ratio(states, actions, target), "status": status,
                "source_density": source_pdf, "target_density": target_pdf}

    def metadata(self):
        return {"name": self.name, "state_coverage": self.state_coverage,
                "action_coverage": self.action_coverage, "state_rare_mass": self.state_rare_mass,
                "action_rare_mass": self.action_rare_mass,
                "state_gap_mass": float(self.state_gap_mass), "action_gap_mass": float(self.action_gap_mass),
                "region_masses": self.region_masses, "density_accounting": "exact",
                "measure": "continuous float64 physical state13 and requested-command4 Lebesgue measure",
                "data_role": "independent controlled terminal queries; not trajectory data",
                "closed_loop_occupancy_known": False}


def nine_cells(state_rare_mass=0.05, action_rare_mass=0.01):
    return tuple(CoverageLaw(s, a, state_rare_mass, action_rare_mass) for s in COVERAGE_LEVELS for a in COVERAGE_LEVELS)


def grouped_split(group_ids, *, fractions=None, seed=0):
    """Split independent units as whole groups; never split trajectories by row.

    This prevents group overlap, but does not make rows within a group IID.
    Conformal calibration below requires one terminal query per independent
    group. All requested partitions receive at least one group.
    """
    groups = _group_ids(group_ids)
    fractions = dict(fractions or {"fit": .4, "selection": .15, "tuning_calibration": .15,
                                   "final_calibration": .15, "evaluation": .15})
    values = np.asarray(list(fractions.values()), dtype=float)
    if not len(values) or not np.isfinite(values).all() or np.any(values <= 0) or not np.isclose(values.sum(), 1.0):
        raise ValueError("positive split fractions must sum to one")
    unique = np.unique(groups)
    if len(unique) < len(values):
        raise ValueError("need at least one independent group per split")
    unique = np.random.default_rng(seed).permutation(unique)
    quotas = values * (len(unique)-len(values))
    counts = np.floor(quotas).astype(int) + 1
    leftovers = len(unique) - counts.sum()
    counts[np.argsort(-(quotas-np.floor(quotas)), kind="stable")[:leftovers]] += 1
    result, start = {}, 0
    for (name, _), count in zip(fractions.items(), counts):
        result[name] = np.flatnonzero(np.isin(groups, unique[start:start+count]))
        start += count
    return result


def assert_disjoint_groups(partitions: Mapping[str, np.ndarray]):
    seen = set()
    for name, groups in partitions.items():
        values = set(_group_ids(groups, allow_empty=True).tolist())
        if seen.intersection(values):
            raise ValueError(f"group leakage involving {name}")
        seen.update(values)


def minimum_equal_weight_calibration(alpha=0.1):
    """Smallest n allowing finite split-conformal rank, with a unit test atom."""
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    value = Decimal(str(alpha))
    return max(1, int(((1-value)/value).to_integral_value(rounding=ROUND_CEILING)))


def binomial_shortfall_probability(n, probability, min_count):
    """P[Binomial(n,p) < min_count], evaluated by a stable log-sum."""
    n, min_count = _integer(n, "n"), _integer(min_count, "min_count")
    p = float(probability)
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("probability must be in [0, 1]")
    if min_count == 0:
        return 0.0
    if n < min_count or p == 0:
        return 1.0
    if p == 1:
        return 0.0
    logs = np.array([math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)
                     + k*math.log(p)+(n-k)*math.log1p(-p) for k in range(min_count)])
    largest = float(logs.max())
    return float(min(1.0, math.exp(largest) * np.exp(logs-largest).sum()))


def plan_binomial_count(min_count, probability, failure_probability=0.05, *, max_samples=10**9):
    """Exact binomial planning for independent group counts in a declared region.

    This plans counts, not weighted conformal rank or an ESS guarantee. A rare
    joint region has product probability; an absent region is impossible.
    """
    min_count = _integer(min_count, "min_count", 1)
    max_samples = _integer(max_samples, "max_samples", min_count)
    p, delta = float(probability), float(failure_probability)
    if not np.isfinite(p) or not 0 <= p <= 1 or not np.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("probability must be in [0,1] and failure_probability in (0,1)")
    common = {"min_count": min_count, "region_probability": p, "failure_probability": delta,
              "unit": "independent groups", "rank_or_ess_guarantee": False}
    if p == 0:
        return dict(common, status="impossible_absent_support", required_groups=None)
    lo, hi = min_count-1, min_count
    while binomial_shortfall_probability(hi, p, min_count) > delta and hi < max_samples:
        hi = min(max_samples, hi*2)
    if binomial_shortfall_probability(hi, p, min_count) > delta:
        return dict(common, status="exceeds_budget", required_groups=None, max_samples=max_samples)
    while hi-lo > 1:
        mid = (lo+hi)//2
        if binomial_shortfall_probability(mid, p, min_count) <= delta:
            hi = mid
        else:
            lo = mid
    return dict(common, status="planned", required_groups=hi,
                achieved_shortfall_probability=binomial_shortfall_probability(hi, p, min_count))


def occupancy_accounting(*, action_policy_known, state_density_method=None):
    """Do not infer a known closed-loop state law from a known action policy."""
    if state_density_method not in (None, "estimated"):
        raise ValueError("this helper cannot certify closed-loop occupancy; use an independently proved law")
    return {"action_policy_known": bool(action_policy_known),
            "state_occupancy": "estimated" if state_density_method else "unknown",
            "joint_occupancy_ratio": "estimated" if state_density_method and action_policy_known else "unknown",
            "exact_weight_guarantee": False}

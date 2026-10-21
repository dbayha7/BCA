"""Exact-mass logging experiments for the finite ThrusterChain companion.

There are two different experiments here. ``fixed_state`` independently resets
to each state/time and samples one action; it is not trajectory data.
``trajectory`` follows the sampled next state for the entire finite horizon.
Retention is a separate, pre-outcome randomized observation process. Dropping a
row never changes the subsequent physical trajectory or the evaluation law.

``Dataset.learner_rows`` is the learner export. ``Dataset.evaluator`` is a
separate audit artifact, including withheld transitions and retention flags.
No value functions, transition kernels, or hidden world labels enter the learner
export. The module uses only NumPy and the standard library.

Exported observations append remaining_fraction=(H-time)/H to the physical
observation. The last transition has next remaining_fraction=0 and
bootstrap_mask=0: truncation ends this finite-horizon objective, so learners
must not bootstrap beyond it even though terminated is False.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np

from .finite import FiniteConfig, FiniteMDP


@dataclass(frozen=True)
class RetentionSpec:
    """Keep ordinary rows; retain critical rows fully, rarely, or never."""

    mode: str = "full"
    rare_probability: float = 0.05

    def __post_init__(self) -> None:
        if self.mode not in ("full", "rare", "zero"):
            raise ValueError("retention mode must be full, rare, or zero")
        if not np.isfinite(self.rare_probability) or not 0 < self.rare_probability < 1:
            raise ValueError("rare_probability must be strictly between zero and one")

    def propensities(self, critical: np.ndarray) -> np.ndarray:
        critical = np.asarray(critical, dtype=bool)
        probability = {"full": 1.0, "rare": self.rare_probability, "zero": 0.0}[self.mode]
        return np.where(critical, probability, 1.0)


@dataclass
class Dataset:
    """Explicit separation between public rows and privileged audit arrays."""

    learner_rows: dict[str, np.ndarray]
    evaluator: dict[str, np.ndarray]
    metadata: dict[str, Any]

    @property
    def n_rows(self) -> int:
        return len(self.learner_rows["state"])

    @property
    def n_complete_rows(self) -> int:
        return len(self.evaluator["state"])


@dataclass
class PairedWorldData:
    """Identical logging experiments plus explicitly separate revealing probes."""

    stall: Dataset
    monotone: Dataset
    probes: dict[str, dict[str, np.ndarray]]
    checks: dict[str, bool]


@dataclass
class RareProbeControl:
    """Mixed-action coupled logs and exact availability of informative probes."""

    stall: Dataset
    monotone: Dataset
    report: dict[str, Any]


# Named streams make the randomization contract inspectable. In particular,
# neither retention nor evaluation changes action or transition random draws.
_NAMESPACES = {"start": 11, "action": 23, "transition": 37, "retention": 53,
               "probe": 71, "optimization": 89}


def _split_code(split: str) -> int:
    """Stable split ID, independent of Python's process-randomized hash."""
    if not isinstance(split, str) or not split:
        raise ValueError("split must be a nonempty string")
    return int.from_bytes(hashlib.sha256(split.encode("utf-8")).digest()[:4], "little")


def _streams(seed: int, split: str) -> tuple[dict[str, np.random.Generator], dict[str, Any]]:
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    split_code = _split_code(split)
    seeds = {name: [int(seed), split_code, code] for name, code in _NAMESPACES.items()}
    generators = {name: np.random.default_rng(np.random.SeedSequence(entropy))
                  for name, entropy in seeds.items() if name != "optimization"}
    provenance = {"generator": "numpy.random.PCG64", "seed_sequence_entropy": seeds,
                  "optimization": "reserved; no learner fitting occurs in data generation"}
    return generators, provenance


def behavior_policy(mdp: FiniteMDP, *, common_only: bool = False,
                    high_action_mass: float = 0.20) -> np.ndarray:
    """Bounded discrete behavior with exact masses, shape ``[H,S,A]``.

    ``high_action_mass`` is the total probability on actions beyond the knee.
    Remaining mass is uniform over common-dynamics actions. Common-only logging
    places exactly zero mass beyond the knee; it does not use an epsilon floor.
    """
    if not np.isfinite(high_action_mass) or not 0 <= high_action_mass <= 1:
        raise ValueError("high_action_mass must lie in [0,1]")
    common = np.asarray(mdp.common_action_mask, dtype=bool)
    if not np.any(common):
        raise ValueError("the model must have common-dynamics actions")
    high = ~common
    mass = 0.0 if common_only or not np.any(high) else float(high_action_mass)
    probabilities = np.zeros(mdp.n_actions, dtype=float)
    probabilities[common] = (1.0 - mass) / np.count_nonzero(common)
    if np.any(high):
        probabilities[high] = mass / np.count_nonzero(high)
    return np.broadcast_to(probabilities, (mdp.horizon, mdp.n_states, mdp.n_actions)).copy()


def _policy(mdp: FiniteMDP, policy: np.ndarray | None, common_only: bool) -> np.ndarray:
    if policy is None:
        return behavior_policy(mdp, common_only=common_only)
    probabilities = np.asarray(policy, dtype=float)
    try:
        probabilities = np.broadcast_to(probabilities,
                                        (mdp.horizon, mdp.n_states, mdp.n_actions)).copy()
    except ValueError as error:
        raise ValueError("policy must broadcast to [H,S,A]") from error
    if (not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(axis=-1), 1.0, rtol=0, atol=1e-12)):
        raise ValueError("policy rows must be nonnegative probability distributions")
    if common_only and np.any(probabilities[..., ~mdp.common_action_mask] > 0):
        raise ValueError("common-only logging cannot assign mass above the knee")
    return probabilities


def _sample_with_uniform(probabilities: np.ndarray, uniform: float) -> int:
    # Set only the terminal CDF endpoint to its mathematical value. This is not
    # support clipping or smoothing: zero-mass outcomes remain impossible.
    # Restrict to positive support first: otherwise correcting a sum a few ulps
    # below one could accidentally put that numerical gap on a zero-mass tail.
    support = np.flatnonzero(probabilities > 0)
    cumulative = np.cumsum(probabilities[support])
    cumulative[-1] = 1.0
    return int(support[np.searchsorted(cumulative, uniform, side="right")])


def _sample(probabilities: np.ndarray, rng: np.random.Generator) -> int:
    return _sample_with_uniform(probabilities, rng.random())


def _finalize(mdp: FiniteMDP, records: list[tuple], *, seed: int, split: str,
              track: str, retention: RetentionSpec, streams: dict,
              seed_provenance: dict, metadata: dict[str, Any]) -> Dataset:
    names = ("episode_id", "time", "state", "action", "next_state", "reward",
             "action_probability", "state_design_probability")
    integer_names = {"episode_id", "time", "state", "action", "next_state"}
    columns = list(zip(*records)) if records else [[] for _ in names]
    rows = {name: np.asarray(column, dtype=np.int64 if name in integer_names else float)
            for name, column in zip(names, columns)}
    count = len(records)
    rows["observation"] = np.column_stack((
        mdp.observations[rows["state"]], (mdp.horizon - rows["time"]) / mdp.horizon))
    rows["next_observation"] = np.column_stack((
        mdp.observations[rows["next_state"]], (mdp.horizon - rows["time"] - 1) / mdp.horizon))
    rows["seed"] = np.full(count, seed, dtype=np.int64)
    rows["provenance"] = np.full(count, "finite:" + track, dtype="U32")
    rows["split"] = np.full(count, split, dtype=f"U{max(1, len(split))}")
    rows["terminated"] = np.zeros(count, dtype=bool)
    rows["truncated"] = rows["time"] == mdp.horizon - 1
    rows["finite_horizon_done"] = rows["truncated"].copy()
    rows["bootstrap_mask"] = (~rows["finite_horizon_done"]).astype(float)
    critical = mdp.critical_mask()[rows["state"], rows["action"]]
    propensities = retention.propensities(critical)
    rows["retention_propensity"] = propensities
    # Critical and retention are determined without consulting reward or next
    # state. A dedicated stream gives nested full/rare/zero retained sets.
    retained = streams["retention"].random(count) < propensities
    learner = {name: value[retained].copy() for name, value in rows.items()}
    evaluator = {name: value.copy() for name, value in rows.items()}
    evaluator["critical"] = np.asarray(critical, dtype=bool)
    evaluator["retained"] = retained
    result_metadata = {
        "schema_version": 1, "track": track, "seed": int(seed), "split": split,
        "seed_provenance": seed_provenance, "retention_mode": retention.mode,
        "rare_probability": retention.rare_probability,
        "complete_row_count": count, "retained_row_count": int(retained.sum()),
        "horizon": mdp.horizon, "physical_observation_dimension": mdp.observations.shape[1],
        "observation_dimension": mdp.observations.shape[1] + 1,
        "observation_schema": (["position", "regime", "load", "remaining_fraction"]
                               if mdp.hidden_world is None else
                               ["position", "load", "remaining_fraction"]),
        "remaining_fraction": "(horizon-time)/horizon; next observation uses time+1",
        "gamma": mdp.gamma, "terminal_continuation": 0.0,
        "objective": "finite-horizon discounted reward with zero terminal continuation",
        "bootstrap_mask_convention": "1 before the horizon; 0 on the final transition despite terminated=False",
        "retention_rule": "pre-outcome critical(state,action); independent Bernoulli draw",
        "retention_measure": "per-opportunity inclusion probability; action probabilities are not renormalized after retention",
        "oracle_arrays_in_learner": False, **metadata,
    }
    return Dataset(learner, evaluator, result_metadata)


def generate_trajectory_data(mdp: FiniteMDP, n_episodes: int = 256, *, seed: int = 0,
                             retention: RetentionSpec | None = None,
                             context_mix: str | np.ndarray = "overlap",
                             policy: np.ndarray | None = None, split: str = "train",
                             common_only: bool = False) -> Dataset:
    """Generate complete physical episodes, then independently retain rows.

    ``state_design_probability`` is NaN for trajectory rows: their state law is
    a time-dependent occupancy, not an independently selected reset design.
    Use the exact MDP occupancy calculation for trajectory importance ratios.
    """
    if not isinstance(n_episodes, (int, np.integer)) or n_episodes < 0:
        raise ValueError("n_episodes must be a nonnegative integer")
    retention = retention or RetentionSpec()
    streams, provenance = _streams(seed, split)
    probabilities = _policy(mdp, policy, common_only)
    initial = mdp.initial_distribution(context_mix)
    records: list[tuple] = []
    for episode in range(n_episodes):
        state = _sample(initial, streams["start"])
        for time in range(mdp.horizon):
            action = _sample(probabilities[time, state], streams["action"])
            successor = _sample(mdp.P[state, action], streams["transition"])
            records.append((episode, time, state, action, successor, mdp.R[state, action],
                            probabilities[time, state, action], np.nan))
            state = successor
    mixture = context_mix if isinstance(context_mix, str) else np.asarray(context_mix).tolist()
    return _finalize(mdp, records, seed=seed, split=split, track="trajectory",
                     retention=retention, streams=streams, seed_provenance=provenance,
                     metadata={"n_episodes": int(n_episodes), "context_mix": mixture,
                               "common_only": common_only,
                               "action_design": "independent categorical conditional on observed state and time"})


def generate_fixed_state_data(mdp: FiniteMDP, samples_per_state_action: int = 4, *,
                              seed: int = 0, retention: RetentionSpec | None = None,
                              split: str = "train", common_only: bool = False) -> Dataset:
    """One-step experiments from every ``(time,state)`` with uniform actions.

    At each fixed state/time, take ``A * samples_per_state_action`` independent
    draws (or the number of allowed actions for common-only logging). Thus the
    argument is an expected sample count per action, not a coverage guarantee.
    The reset design is uniform over state/time strata. These rows must never be
    interpreted as consecutive transitions of a trajectory.
    """
    if (not isinstance(samples_per_state_action, (int, np.integer))
            or samples_per_state_action < 0):
        raise ValueError("samples_per_state_action must be a nonnegative integer")
    retention = retention or RetentionSpec()
    streams, provenance = _streams(seed, split)
    allowed = np.asarray(mdp.common_action_mask if common_only else
                         np.ones(mdp.n_actions, dtype=bool))
    probabilities = allowed.astype(float) / allowed.sum()
    n_per_stratum = int(allowed.sum()) * samples_per_state_action
    records: list[tuple] = []
    for time in range(mdp.horizon):
        for state in range(mdp.n_states):
            for _ in range(n_per_stratum):
                action = _sample(probabilities, streams["action"])
                successor = _sample(mdp.P[state, action], streams["transition"])
                records.append((len(records), time, state, action, successor,
                                mdp.R[state, action], probabilities[action],
                                1.0 / (mdp.horizon * mdp.n_states)))
    return _finalize(mdp, records, seed=seed, split=split, track="fixed_state",
                     retention=retention, streams=streams, seed_provenance=provenance,
                     metadata={"samples_per_state_action_expected": int(samples_per_state_action),
                               "common_only": common_only,
                               "action_design": "independent uniform categorical at each fixed state/time",
                               "state_design": "balanced fixed state/time strata; joint stratum mass 1/(H*S)"})


def generate_evaluation_data(mdp: FiniteMDP, n_episodes: int = 256, *, seed: int = 1,
                             context_mix: str | np.ndarray = "overlap",
                             policy: np.ndarray | None = None) -> Dataset:
    """Independent, unfiltered evaluation episodes in a reserved split.

    The split namespace makes the evaluation draws independent of train draws
    even when the same root seed is deliberately supplied to both generators.
    Evaluation retains all critical rows regardless of any training retention.
    """
    return generate_trajectory_data(mdp, n_episodes, seed=seed, context_mix=context_mix,
                                    policy=policy, split="evaluation", retention=RetentionSpec("full"))


def _identical_rows(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> bool:
    if left.keys() != right.keys():
        return False
    for name in left:
        a, b = left[name], right[name]
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if np.issubdtype(a.dtype, np.inexact):
            if not np.array_equal(a, b, equal_nan=True):
                return False
        elif not np.array_equal(a, b):
            return False
    return True


def paired_hidden_control(config: FiniteConfig | None = None, *, n_episodes: int = 64,
                          seed: int = 0, n_probe_rows: int = 16) -> PairedWorldData:
    """Coupled hidden worlds with a kernel-level nonidentifiability certificate.

    Logging takes only common-dynamics actions. Observations, public metadata,
    and complete logs have no world identifier and are identical under coupling.
    The returned small high-action probes are *separate evaluator-only data*;
    they are never merged into either learner export.
    """
    if not isinstance(n_probe_rows, (int, np.integer)) or n_probe_rows < 1:
        raise ValueError("n_probe_rows must be a positive integer")
    config = config or FiniteConfig()
    stall_mdp = FiniteMDP(config, hidden_world="stall")
    monotone_mdp = FiniteMDP(config, hidden_world="monotone")
    common = stall_mdp.common_action_mask
    policy = behavior_policy(stall_mdp, common_only=True)
    initial = stall_mdp.initial_distribution("overlap")
    reachable = initial > 0
    reached = reachable.copy()
    reachable_kernel_equal = True
    reachable_reward_equal = True
    for time in range(stall_mdp.horizon):
        support = reachable[:, None] & (policy[time] > 0)
        reachable_kernel_equal &= np.array_equal(stall_mdp.P[support], monotone_mdp.P[support])
        reachable_reward_equal &= np.array_equal(stall_mdp.R[support], monotone_mdp.R[support])
        reachable = np.any(stall_mdp.P[support] > 0, axis=0)
        reached |= reachable
    stall = generate_trajectory_data(stall_mdp, n_episodes, seed=seed, split="paired_log",
                                     common_only=True)
    monotone = generate_trajectory_data(monotone_mdp, n_episodes, seed=seed, split="paired_log",
                                        common_only=True)
    streams, _ = _streams(seed, "paired_probes")
    # Select probes from pre-outcome critical states/actions, not from observed
    # successes or failures. These are interventions, not the logging policy.
    candidate_pairs = np.argwhere(stall_mdp.critical_mask())
    if len(candidate_pairs) == 0:
        candidate_pairs = np.argwhere(np.broadcast_to(~common, (stall_mdp.n_states, stall_mdp.n_actions)))
    selected = candidate_pairs[streams["probe"].integers(0, len(candidate_pairs), n_probe_rows)]
    uniforms = streams["transition"].random(n_probe_rows)
    probes: dict[str, dict[str, np.ndarray]] = {}
    for name, mdp in (("stall", stall_mdp), ("monotone", monotone_mdp)):
        states, actions = selected.T
        successors = np.asarray([_sample_with_uniform(mdp.P[state, action], uniform)
                                 for (state, action), uniform in zip(selected, uniforms)], dtype=np.int64)
        action_counts = np.bincount(candidate_pairs[:, 0], minlength=mdp.n_states)
        probes[name] = {"episode_id": np.arange(n_probe_rows, dtype=np.int64),
                        "time": np.zeros(n_probe_rows, dtype=np.int64),
                        "state": states.copy(), "action": actions.copy(),
                        "observation": np.column_stack((mdp.observations[states], np.ones(n_probe_rows))),
                        "next_state": successors,
                        "next_observation": np.column_stack((mdp.observations[successors],
                                                              np.full(n_probe_rows, (mdp.horizon - 1) / mdp.horizon))),
                        "reward": mdp.R[states, actions].copy(),
                        "action_probability": 1.0 / action_counts[states],
                        "state_design_probability": action_counts[states] / len(candidate_pairs),
                        "retention_propensity": np.ones(n_probe_rows),
                        "provenance": np.full(n_probe_rows, "finite:paired_probe"),
                        "terminated": np.zeros(n_probe_rows, dtype=bool),
                        "truncated": np.full(n_probe_rows, mdp.horizon == 1),
                        "finite_horizon_done": np.full(n_probe_rows, mdp.horizon == 1),
                        "bootstrap_mask": np.full(n_probe_rows, float(mdp.horizon != 1)),
                        "transition_probability": mdp.P[states, actions].copy(),
                        "split": np.full(n_probe_rows, "paired_probes"),
                        "seed": np.full(n_probe_rows, seed, dtype=np.int64)}
    checks = {
        "initial_law_equal": np.array_equal(initial, monotone_mdp.initial_distribution("overlap")),
        "observation_map_equal": np.array_equal(stall_mdp.observations, monotone_mdp.observations),
        "common_kernel_equal": np.array_equal(stall_mdp.P[:, common], monotone_mdp.P[:, common]),
        "common_reward_equal": np.array_equal(stall_mdp.R[:, common], monotone_mdp.R[:, common]),
        "reachable_kernel_equal": bool(reachable_kernel_equal),
        "reachable_reward_equal": bool(reachable_reward_equal),
        "complete_logs_equal": _identical_rows(stall.evaluator, monotone.evaluator),
        "learner_logs_equal": _identical_rows(stall.learner_rows, monotone.learner_rows),
        "learner_metadata_equal": stall.metadata == monotone.metadata,
        "logging_actions_common_only": bool(np.all(common[stall.evaluator["action"]])),
        "hidden_observation_dimension": (stall_mdp.observations.shape[1] == 2
                                         and stall.learner_rows["observation"].shape[1] == 3),
        "separate_probes_reveal_kernel_difference": not np.array_equal(
            probes["stall"]["transition_probability"], probes["monotone"]["transition_probability"]),
    }
    return PairedWorldData(stall, monotone, probes, checks)


def probe_identifiability(stall_mdp: FiniteMDP, monotone_mdp: FiniteMDP,
                          policy: np.ndarray, *, n_episodes: int = 1,
                          initial: np.ndarray | None = None) -> dict[str, Any]:
    """Exact no-informative-action probability by a substochastic kernel DP.

    A state/action is informative here iff its two next-state kernels or rewards
    differ. Remove such actions and propagate the remaining probability mass.
    The surviving mass is an identical component of the two complete log laws.
    Its complement measures probe availability, not guaranteed identification:
    differing kernels can still produce the same realized next observation.
    """
    if not isinstance(n_episodes, (int, np.integer)) or n_episodes < 0:
        raise ValueError("n_episodes must be a nonnegative integer")
    if (stall_mdp.P.shape != monotone_mdp.P.shape
            or stall_mdp.horizon != monotone_mdp.horizon
            or not np.array_equal(stall_mdp.observations, monotone_mdp.observations)):
        raise ValueError("paired models must have matching states, horizon, and observation maps")
    probabilities = _policy(stall_mdp, policy, common_only=False)
    initial = (stall_mdp.initial_distribution("overlap") if initial is None
               else np.asarray(initial, dtype=float))
    if (initial.shape != (stall_mdp.n_states,) or np.any(initial < 0)
            or not np.all(np.isfinite(initial))
            or not np.isclose(initial.sum(), 1.0, rtol=0, atol=1e-12)):
        raise ValueError("initial must be a probability distribution on paired states")
    different = (np.any(stall_mdp.P != monotone_mdp.P, axis=-1)
                 | (stall_mdp.R != monotone_mdp.R))
    safe = ~different
    per_world: dict[str, dict[str, float]] = {}
    for name, mdp in (("stall", stall_mdp), ("monotone", monotone_mdp)):
        survival = initial.copy()
        for time in range(mdp.horizon):
            safe_state_action_mass = survival[:, None] * probabilities[time] * safe
            survival = np.einsum("sa,sak->k", safe_state_action_mass, mdp.P)
        no_hit = float(survival.sum())
        # Enforce the exact no-probe boundary. The forward recursion may sum to
        # 1 +/- a few ulps; this branch is justified by the empty event support.
        if not np.any(probabilities * different[None, :, :] > 0):
            no_hit = 1.0
        if not -1e-12 <= no_hit <= 1 + 1e-12:
            raise ArithmeticError("substochastic propagation produced invalid probability")
        no_hit = min(1.0, max(0.0, no_hit))  # floating-point roundoff only
        all_no_hit = no_hit ** int(n_episodes)
        per_world[name] = {
            "no_differing_kernel_hit_probability_per_episode": no_hit,
            "no_differing_kernel_hit_probability_all_episodes": all_no_hit,
            "at_least_one_differing_kernel_hit_probability": 1.0 - all_no_hit,
        }
    safe_kernel_stall = stall_mdp.P * safe[..., None]
    safe_kernel_monotone = monotone_mdp.P * safe[..., None]
    return {
        "n_episodes": int(n_episodes), "horizon": stall_mdp.horizon,
        "differing_kernel_state_action_count": int(np.count_nonzero(different)),
        "safe_kernel_equal": bool(np.array_equal(safe_kernel_stall, safe_kernel_monotone)),
        "per_world": per_world,
        "meaning": "Probability of selecting a state/action with a different paired kernel or reward; not guaranteed statistical identification.",
        "method": "Finite-horizon propagation through the substochastic kernel after removing differing-kernel state/actions; independent-episode exponentiation.",
    }


def rare_probe_control(config: FiniteConfig | None = None, *, n_episodes: int = 64,
                       seed: int = 0, probe_probability: float = 0.01) -> RareProbeControl:
    """Log actual hidden-world trajectories with a small high-action probability.

    This is a separate experiment from the common-only negative control.
    Coupling uses identical start, action, and transition uniforms but trajectories
    may diverge after an informative probe. No world identifier enters public
    rows or metadata. All rows are retained, including probes.
    """
    config = config or FiniteConfig()
    stall_mdp = FiniteMDP(config, hidden_world="stall")
    monotone_mdp = FiniteMDP(config, hidden_world="monotone")
    policy = behavior_policy(stall_mdp, high_action_mass=probe_probability)
    report = probe_identifiability(stall_mdp, monotone_mdp, policy,
                                   n_episodes=n_episodes)
    common_only = probe_probability == 0
    stall = generate_trajectory_data(stall_mdp, n_episodes, seed=seed, policy=policy,
                                     common_only=common_only, split="rare_probe_log")
    monotone = generate_trajectory_data(monotone_mdp, n_episodes, seed=seed, policy=policy,
                                        common_only=common_only, split="rare_probe_log")
    different = (np.any(stall_mdp.P != monotone_mdp.P, axis=-1)
                 | (stall_mdp.R != monotone_mdp.R))
    for name, dataset in (("stall", stall), ("monotone", monotone)):
        dataset.metadata["probe_probability"] = float(probe_probability)
        rows = dataset.evaluator
        hits = different[rows["state"], rows["action"]]
        rows["differing_kernel_action"] = hits
        report["per_world"][name]["observed_differing_kernel_action_count"] = int(hits.sum())
        report["per_world"][name]["observed_episodes_with_differing_kernel_action"] = int(
            np.unique(rows["episode_id"][hits]).size)
    report.update({
        "probe_probability": float(probe_probability),
        "seed": int(seed), "split": "rare_probe_log",
        "common_support_kernel_equal": bool(np.array_equal(
            stall_mdp.P[:, stall_mdp.common_action_mask],
            monotone_mdp.P[:, monotone_mdp.common_action_mask])),
        "zero_probe_boundary": common_only,
        "coupled_complete_logs_equal": _identical_rows(stall.evaluator, monotone.evaluator),
        "learner_metadata_equal": stall.metadata == monotone.metadata,
        "probability_no_high_action_per_episode": (1.0 - probe_probability) ** config.horizon,
        "probability_no_high_action_all_episodes": (1.0 - probe_probability) ** (config.horizon * n_episodes),
    })
    return RareProbeControl(stall, monotone, report)

"""Budgeted adapter training and separate, cooperative frozen evaluation.

The runner gives learners only immutable train transitions. Its checkpoints bind
the adapter state and offline minibatch RNG, so offline continuation is exact.
The evaluation guard is Python instrumentation, not a security sandbox.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .datasets import TrajectoryDataset, assert_training_allowed, canonical_json
from .inference_guard import InferenceGuard
from .training_protocol import (AdapterCapabilities, EvaluationConfig,
                                LearnerCheckpoint, OfflineBudget, TransitionBatch)


FAMILIES = ("recorded", "strong_current", "big_waves", "big_kicks")
STREAMS = {"offline_batches": 7101, "online_offline_batches": 7102,
           "online_replay_batches": 7103, "adaptation_resets": 7104,
           "evaluation_resets": 7105, "learner_initialization": 7106}


class RunnerError(ValueError):
    """A learner or run does not satisfy the declared execution contract."""


def _hash(value):
    return hashlib.sha256(canonical_json(value).encode("utf8")).hexdigest()


def _seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise RunnerError("seed must be a nonnegative integer")
    return int(seed)


def _rng(seed, name, family=None):
    key = (STREAMS[name],) if family is None else (STREAMS[name], FAMILIES.index(family))
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(_seed(seed), spawn_key=key)))


def _settings(value):
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RunnerError("learning_settings must be a JSON mapping")
    import json
    return json.loads(canonical_json(dict(value)))


def _dataset(dataset):
    if type(dataset) is not TrajectoryDataset:
        raise RunnerError("runner requires a validated immutable TrajectoryDataset")
    return dataset.metadata


def train_transitions(dataset):
    """Expose only train observations and executed actions to an adapter."""
    _dataset(dataset)
    split = dataset.split("train")
    observations = split["observations"]
    horizon = split["actions"].shape[1]
    terminated = split["terminated"].reshape(-1)
    return TransitionBatch(
        observations=observations[:, :-1].reshape(-1, observations.shape[-1]),
        actions=split["actions"].reshape(-1, 4),
        rewards=split["rewards"].reshape(-1),
        next_observations=observations[:, 1:].reshape(-1, observations.shape[-1]),
        terminated=terminated, truncated=split["truncated"].reshape(-1),
        bootstrap_mask=(~terminated).astype(np.float64),
        origin=np.zeros(terminated.size, dtype=np.uint8),
        group_ids=np.repeat(split["group_ids"], horizon))


def _capabilities(learner, *, online=False):
    capabilities = learner.capabilities
    if not isinstance(capabilities, AdapterCapabilities):
        raise RunnerError("learner must expose AdapterCapabilities")
    if capabilities.fit_semantics != "optimizer_steps":
        raise RunnerError("closed-form adapters do not support gradient-update budgets; use their existing fitting path")
    if not isinstance(capabilities.primary_optimizer, str) or not capabilities.primary_optimizer:
        raise RunnerError("optimizer adapters must name their primary optimizer")
    if online and (not capabilities.supports_online or not callable(
            getattr(learner, "configure_learning_settings", None))):
        raise RunnerError("adapter does not support online learning and explicit learning settings")
    return capabilities


def _checkpoint(learner):
    checkpoint = learner.checkpoint()
    if not isinstance(checkpoint, LearnerCheckpoint):
        raise RunnerError("checkpoint() must return a LearnerCheckpoint")
    if checkpoint.adapter_id != learner.capabilities.adapter_id:
        raise RunnerError("checkpoint adapter identity mismatch")
    if checkpoint.primary_optimizer != learner.capabilities.primary_optimizer:
        raise RunnerError("checkpoint primary optimizer differs from adapter capability")
    counters = _counters(learner)
    if counters.optimizer_steps != checkpoint.optimizer_steps or counters.optimizer_step_counts != dict(checkpoint.optimizer_step_counts):
        raise RunnerError("lightweight optimizer counters disagree with checkpoint")
    return checkpoint


@dataclass(frozen=True)
class _Counters:
    optimizer_steps: int
    optimizer_step_counts: dict


def _counters(learner):
    total = learner.optimizer_steps
    counts = dict(learner.optimizer_step_counts)
    if type(total) is not int or total < 0 or any(type(value) is not int or value < 0 for value in counts.values()):
        raise RunnerError("optimizer counters must be nonnegative Python integers")
    if any(not isinstance(name, str) or not name for name in counts) or sum(counts.values()) != total:
        raise RunnerError("per-optimizer counters disagree with total optimizer steps")
    if learner.capabilities.primary_optimizer not in counts:
        raise RunnerError("primary optimizer is absent from optimizer counters")
    return _Counters(total, counts)


def _counts_delta(before, after):
    names = set(before.optimizer_step_counts) | set(after.optimizer_step_counts)
    delta = {name: after.optimizer_step_counts.get(name, 0) - before.optimizer_step_counts.get(name, 0)
             for name in sorted(names)}
    if any(value < 0 for value in delta.values()):
        raise RunnerError("optimizer counters moved backwards")
    if sum(delta.values()) != after.optimizer_steps - before.optimizer_steps:
        raise RunnerError("per-optimizer counters disagree with total optimizer steps")
    return delta


def _update(learner, batch, before):
    assert_training_allowed("training_runner.update")
    metrics = learner.update(batch)
    after = _counters(learner)
    _counts_delta(before, after)
    primary = learner.capabilities.primary_optimizer
    if after.optimizer_step_counts[primary] - before.optimizer_step_counts[primary] != 1:
        raise RunnerError("update(batch) must execute exactly one primary optimizer step; mismatch or overshoot refused")
    if not isinstance(metrics, dict):
        raise RunnerError("update(batch) must return a metrics dictionary")
    # Validate before admitting metrics into a portable receipt.
    _hash(metrics)
    return after, metrics


def _restore(learner, checkpoint):
    if checkpoint.adapter_id != learner.capabilities.adapter_id:
        raise RunnerError("adapter factory does not match pretrained checkpoint")
    learner.restore(checkpoint)
    restored = _checkpoint(learner)
    if restored.digest != checkpoint.digest:
        raise RunnerError("restore did not reproduce complete checkpoint state and counters")
    return restored


def _with_runner_state(checkpoint, state):
    metadata = checkpoint.metadata
    metadata["runner_state"] = state
    return LearnerCheckpoint(adapter_id=checkpoint.adapter_id,
                             components=dict(checkpoint.components),
                             optimizer_steps=checkpoint.optimizer_steps,
                             optimizer_step_counts=dict(checkpoint.optimizer_step_counts),
                             metadata=metadata)


def _identity(dataset):
    metadata = dataset.metadata
    return {"dataset_sha256": dataset.digest,
            "split_sha256": {"train": dataset.split_digest("train")},
            "preprocessing_source": "train_only", "learner_visible_split": "train",
            "provenance": metadata["provenance"],
            "environment_config_sha256": _hash(metadata["source_model_metadata"]["environment_config"])}


def train_offline(dataset, factory, *, budget=OfflineBudget(), seed=20361022,
                  learning_settings=None, resume_from=None):
    """Execute exactly ``budget.updates`` additional primary optimizer steps.

    ``resume_from`` must be a checkpoint returned by this runner. The same
    dataset, seed, batch size and settings are required; its stored PCG64 state
    continues the minibatch sequence rather than restarting that sequence.
    No environment is constructed or stepped by this function.
    """
    assert_training_allowed("train_offline")
    if not isinstance(budget, OfflineBudget):
        raise RunnerError("budget must be OfflineBudget")
    _dataset(dataset)
    seed = _seed(seed)
    settings = _settings(learning_settings)
    learner = factory()
    capabilities = _capabilities(learner)
    identity = _identity(dataset)
    batch = train_transitions(dataset)
    sampler = _rng(seed, "offline_batches")
    resumed_updates = 0
    if resume_from is None:
        init_seed = int(_rng(seed, "learner_initialization").integers(0, 2**63 - 1))
        learner.initialize(batch, seed=init_seed, learning_settings=_settings(settings))
        before = _checkpoint(learner)
        if before.optimizer_steps != 0 or any(before.optimizer_step_counts.values()):
            raise RunnerError("initialize must not execute optimizer steps")
    else:
        if not isinstance(resume_from, LearnerCheckpoint):
            raise RunnerError("resume_from must be a LearnerCheckpoint")
        state = resume_from.metadata.get("runner_state", {})
        required = {"phase": "offline", "dataset_sha256": dataset.digest,
                    "train_split_sha256": identity["split_sha256"]["train"],
                    "seed": seed, "batch_size": budget.batch_size,
                    "learning_settings": settings}
        if any(state.get(key) != value for key, value in required.items()):
            raise RunnerError("offline resume requires matching dataset, seed, batch size and learning settings")
        if "sampler_rng_state" not in state or "total_runner_updates" not in state:
            raise RunnerError("offline checkpoint is missing runner sampling state")
        sampler.bit_generator.state = state["sampler_rng_state"]
        resumed_updates = state["total_runner_updates"]
        if (type(resumed_updates) is not int or resumed_updates < 0 or resumed_updates !=
                resume_from.optimizer_step_counts.get(capabilities.primary_optimizer)):
            raise RunnerError("offline resume cursor must equal the checkpoint primary optimizer counter")
        if resumed_updates + budget.updates > 3_000_000:
            raise RunnerError("cumulative offline updates cannot exceed 3000000")
        before = _restore(learner, resume_from)
    if resumed_updates + budget.updates > 3_000_000:
        raise RunnerError("cumulative offline updates cannot exceed 3000000")
    last, metrics = _counters(learner), {}
    for _ in range(budget.updates):
        indices = sampler.integers(0, len(batch), size=budget.batch_size)
        last, metrics = _update(learner, batch.select(indices), last)
    last = _checkpoint(learner)
    primary = capabilities.primary_optimizer
    executed_primary = last.optimizer_step_counts[primary] - before.optimizer_step_counts[primary]
    executed_total = last.optimizer_steps - before.optimizer_steps
    if executed_primary != budget.updates:
        raise RunnerError("offline executed optimizer steps differ from requested budget")
    state = {"phase": "offline", "dataset_sha256": dataset.digest,
             "train_split_sha256": identity["split_sha256"]["train"], "seed": seed,
             "batch_size": budget.batch_size, "learning_settings": settings,
             "sampler_rng_state": sampler.bit_generator.state,
             "total_runner_updates": resumed_updates + budget.updates}
    checkpoint = _with_runner_state(last, state)
    config = {"budget": budget.to_dict(), "seed": seed, "learning_settings": settings}
    receipt = {"schema": "thrusterchain.offline_training.v4", **identity,
               "adapter_id": capabilities.adapter_id, "fit_semantics": capabilities.fit_semantics,
               "primary_optimizer": primary,
               "requested_primary_optimizer_updates": budget.updates,
               "status": "trained" if budget.updates else "initialized_no_updates",
               "config": config, "config_sha256": _hash(config),
               "effective_learning_settings": _effective_settings(checkpoint),
               "effective_learning_settings_sha256": _hash(_effective_settings(checkpoint)),
               "budget": {"source": "explicit_offline_optimizer_budget",
                          "requested": budget.to_dict(), "effective": budget.to_dict(),
                          "executed": {"updates": executed_primary,
                                       "primary_optimizer_updates": executed_primary,
                                       "optimizer_steps_total": executed_total,
                                       "batch_size": budget.batch_size, "environment_interactions": 0}},
               "optimizer_steps_before": before.optimizer_steps, "optimizer_steps_after": last.optimizer_steps,
               "executed_primary_optimizer_updates": executed_primary,
               "executed_optimizer_steps_total": executed_total,
               "optimizer_step_counts": dict(last.optimizer_step_counts),
               "optimizer_step_deltas": _counts_delta(before, last),
               "checkpoint_sha256": checkpoint.digest,
               "resumed_from_sha256": None if resume_from is None else resume_from.digest,
               "resume_semantics": "additional updates; full adapter state plus offline PCG64 sampler restored",
               "sampling": {"bit_generator": "PCG64", "with_replacement": True,
                            "stream": "offline_batches", "stream_key": STREAMS["offline_batches"]},
               "environment_interactions": 0, "last_update_metrics": metrics}
    receipt["receipt_sha256"] = _hash(receipt)
    return checkpoint, receipt


def _fresh_seed(rng, forbidden):
    while True:
        value = int(rng.integers(0, 2**63 - 1))
        if value not in forbidden:
            forbidden.add(value)
            return value


def _validate_families(families):
    families = tuple(families)
    if not families or len(set(families)) != len(families) or any(f not in FAMILIES for f in families):
        raise RunnerError("families must be unique canonical start families")
    return families


def _checkpoint_destination(directory):
    """Preflight without creating outputs or collecting any interactions."""
    if directory is None:
        return None
    destination = Path(directory).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"checkpoint directory already exists: {destination}")
    for ancestor in destination.parents:
        if ancestor.exists() or ancestor.is_symlink():
            if not ancestor.is_dir():
                raise NotADirectoryError(f"checkpoint ancestor is not a directory: {ancestor}")
            break
    return destination


def _policy_state(policy):
    result = policy.state_bytes()
    if not isinstance(result, bytes):
        raise RunnerError("frozen policy state_bytes() must serialize all inference state as bytes")
    return result


def _command(action):
    raw = np.asarray(action)
    if raw.dtype.kind not in "fiu" or raw.shape != (4,) or not np.isfinite(raw).all():
        raise RunnerError("policy actions must be four finite real numbers")
    requested = raw.astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        deployed = requested.astype(np.float32)
    if not np.isfinite(deployed).all():
        raise RunnerError("policy action is nonfinite at float32 deployment")
    return requested, np.clip(deployed, -1, 1)


def _end_flags(step, terminated, truncated, info, semantics):
    done = step == 99
    if bool(terminated) != (done and semantics == "finite_horizon") or bool(truncated) != (
            done and semantics == "continuing_time_limit"):
        raise RunnerError("canonical episode terminal flags or length changed")
    expected = 0.0 if done and semantics == "finite_horizon" else 1.0
    if info.get("bootstrap_mask") != expected:
        raise RunnerError("canonical bootstrap semantics changed")


def evaluate_frozen(policy, *, family, episodes, episode_seeds, physics,
                    task_semantics, provenance="smoke"):
    """Complete fixed-policy episodes with a state check around every action."""
    _validate_families((family,))
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
        raise RunnerError("frozen evaluation requires positive episodes")
    episode_seeds = tuple(_seed(seed) for seed in episode_seeds)
    if len(episode_seeds) != episodes or len(set(episode_seeds)) != episodes:
        raise RunnerError("each frozen episode requires its own seed")
    from .environment import ThrusterChainEnv
    records, inference_calls = [], 0
    with InferenceGuard() as guard:
        initial = _policy_state(policy)
        env = ThrusterChainEnv(**physics, start=family, observation_mode="time_aware", task_semantics=task_semantics)
        try:
            for episode_seed in episode_seeds:
                obs, _ = env.reset(seed=int(episode_seed))
                total, discounted, bootstrap_masks, terminal_flags, truncated_flags = 0.0, 0.0, [], [], []
                for step in range(100):
                    before = _policy_state(policy)
                    if before != initial:
                        raise RunnerError("frozen policy state changed before action")
                    action = policy.act(obs.copy())
                    inference_calls += 1
                    if _policy_state(policy) != before:
                        raise RunnerError("frozen policy state changed during action")
                    requested, expected_command = _command(action)
                    obs, reward, terminated, truncated, info = env.step(requested)
                    if not np.array_equal(np.asarray(info["command"], dtype=np.float32), expected_command):
                        raise RunnerError("executed action mapping differs from canonical deployment")
                    _end_flags(step, terminated, truncated, info, task_semantics)
                    total += float(reward)
                    discounted += .98**step * float(reward)
                    bootstrap_masks.append(float(info["bootstrap_mask"]))
                    terminal_flags.append(bool(terminated))
                    truncated_flags.append(bool(truncated))
                records.append({"seed": int(episode_seed), "steps": 100, "complete": True,
                                "return": total, "discounted_return": discounted,
                                "bootstrap_masks": bootstrap_masks, "terminated": terminal_flags,
                                "truncated": truncated_flags})
        finally:
            env.close()
        final_state = _policy_state(policy)
        if final_state != initial:
            raise RunnerError("frozen policy state changed during evaluation")
    audit = guard.receipt()
    if any(audit[key] for key in ("training_calls", "dataset_collection_calls", "write_attempts", "forbidden_api_calls")):
        raise RunnerError("forbidden operation attempted during frozen evaluation")
    returns = [record["return"] for record in records]
    return {"schema": "thrusterchain.adapter_frozen_evaluation.v4", "family": family,
            "episodes": episodes, "episode_seeds": [int(seed) for seed in episode_seeds],
            "episode_records": records, "returns": returns,
            "discounted_returns": [record["discounted_return"] for record in records],
            "mean_return": float(np.mean(returns)),
            "sample_sd_return": float(np.std(returns, ddof=1)) if episodes > 1 else None,
            "environment_steps": episodes * 100, "parameter_updates": 0,
            "normalization_updates": 0, "inference_calls": inference_calls,
            "policy_sha256_before": hashlib.sha256(initial).hexdigest(),
            "policy_sha256_after": hashlib.sha256(final_state).hexdigest(),
            "inference_audit": audit, "provenance": provenance,
            "evidence_status": "smoke_only_not_research_evidence" if provenance == "smoke" else "single_frozen_policy_run",
            "held_out_scenario_family": family != "recorded"}


def _replay_batch(rows, indices):
    selected = [rows[int(index)] for index in indices]
    return TransitionBatch(**{name: np.asarray([row[name] for row in selected])
                              for name in ("observations", "actions", "rewards", "next_observations",
                                           "terminated", "truncated", "bootstrap_mask", "origin", "group_ids")})


def _configure_online(learner, settings):
    before = _checkpoint(learner)
    learner.configure_learning_settings(settings)
    after = _checkpoint(learner)
    if before.optimizer_steps != after.optimizer_steps or dict(before.optimizer_step_counts) != dict(after.optimizer_step_counts):
        raise RunnerError("configuring learning settings changed optimizer counters")
    for name in before.components:
        # Opaque optimizer bytes may include hyperparameters. The adapter
        # contract requires preserving momentum and all non-setting state.
        if name == "optimizer":
            continue
        if before.components[name] != after.components.get(name):
            raise RunnerError("configuring learning settings changed checkpoint component " + name)
    return before, after


def _effective_settings(checkpoint):
    metadata = checkpoint.metadata
    return metadata.get("adapter", {}).get("learning_settings", metadata.get("learning_settings", {}))


def _adapt(dataset, learner, checkpoint, config, family, forbidden):
    from .environment import ThrusterChainEnv
    online = config.online
    offline = train_transitions(dataset)
    physics = dataset.metadata["physics"]
    semantics = dataset.metadata["task_semantics"]
    rng_offline = _rng(config.seed, "online_offline_batches", family)
    rng_online = _rng(config.seed, "online_replay_batches", family)
    rng_reset = _rng(config.seed, "adaptation_resets", family)
    n_offline = math.floor(online.batch_size * online.offline_fraction)
    n_online = online.batch_size - n_offline
    if online.updates:
        setting_before, last = _configure_online(learner, _settings(online.learning_settings))
    else:
        setting_before = last = _checkpoint(learner)
    configured_checkpoint = last
    rows, complete, reset_seeds = [], [], []
    total, discounted, episode_step, episode = 0.0, 0.0, 0, 0
    sampled_offline, sampled_online, metrics = 0, 0, {}
    env = ThrusterChainEnv(**physics, start=family, observation_mode="time_aware", task_semantics=semantics)
    try:
        reset_seed = _fresh_seed(rng_reset, forbidden)
        reset_seeds.append(reset_seed)
        obs, _ = env.reset(seed=reset_seed)
        for interaction in range(online.interactions):
            # Exploration may advance adapter RNG, but must never hide updates.
            action_before = _counters(learner)
            requested, expected_command = _command(learner.act(obs.copy(), explore=True))
            action_after = _counters(learner)
            if action_before.optimizer_steps != action_after.optimizer_steps or dict(
                    action_before.optimizer_step_counts) != dict(action_after.optimizer_step_counts):
                raise RunnerError("act(explore=True) executed undeclared optimizer steps")
            nxt, reward, terminated, truncated, info = env.step(requested)
            executed = np.asarray(info["command"], dtype=np.float32)
            if not np.array_equal(executed, expected_command):
                raise RunnerError("executed action mapping differs from canonical deployment")
            _end_flags(episode_step, terminated, truncated, info, semantics)
            rows.append({"observations": obs.copy(), "actions": executed.copy(),
                         "requested_actions": requested.copy(), "rewards": float(reward),
                         "next_observations": nxt.copy(), "terminated": bool(terminated),
                         "truncated": bool(truncated), "bootstrap_mask": float(info["bootstrap_mask"]),
                         "origin": 1, "group_ids": -(episode + 1), "episode": episode,
                         "episode_step": episode_step, "interaction": interaction + 1})
            total += float(reward)
            discounted += .98**episode_step * float(reward)
            episode_step += 1
            obs = nxt
            last = action_after
            updates_now = ((interaction + 1) * online.updates // online.interactions
                           - interaction * online.updates // online.interactions)
            for _ in range(updates_now):
                mixed = None
                if n_offline:
                    mixed = offline.select(rng_offline.integers(0, len(offline), size=n_offline))
                if n_online:
                    live = _replay_batch(rows, rng_online.integers(0, len(rows), size=n_online))
                    mixed = live if mixed is None else mixed.concat(live)
                last, metrics = _update(learner, mixed, last)
                sampled_offline += n_offline
                sampled_online += n_online
            if terminated or truncated:
                complete.append({"episode": episode, "seed": reset_seed, "steps": episode_step,
                                 "return": total, "discounted_return": discounted, "complete": True})
                episode += 1
                total, discounted, episode_step = 0.0, 0.0, 0
                if interaction + 1 < online.interactions:
                    reset_seed = _fresh_seed(rng_reset, forbidden)
                    reset_seeds.append(reset_seed)
                    obs, _ = env.reset(seed=reset_seed)
    finally:
        env.close()
    final = _checkpoint(learner)
    primary = learner.capabilities.primary_optimizer
    primary_delta = final.optimizer_step_counts[primary] - checkpoint.optimizer_step_counts[primary]
    if len(rows) != online.interactions or primary_delta != online.updates:
        raise RunnerError("online interactions or optimizer steps differ from requested budget")
    if online.updates == 0 and any(final.components[name] != checkpoint.components[name]
                                  for name in checkpoint.components if name != "rng"):
        raise RunnerError("collection without updates changed non-RNG learner state")
    final = _with_runner_state(final, {"phase": "online_end_checkpoint",
            "dataset_sha256": dataset.digest, "train_split_sha256": dataset.split_digest("train"),
            "pretrained_checkpoint_sha256": checkpoint.digest, "family": family,
            "online_config": online.to_dict(), "seed": config.seed,
            "resume_semantics": "learner continuation only; replay and environment cursor not serialized"})
    _restore(learner, final)
    partial = None if episode_step == 0 else {"episode": episode, "seed": reset_seed,
                "steps": episode_step, "return": total, "discounted_return": discounted, "complete": False}
    # JSON records retain requested commands separately from replay's executed action.
    raw_rows = [{name: value.tolist() if isinstance(value, np.ndarray) else value for name, value in row.items()}
                for row in rows]
    raw = {"status": "adapted" if online.updates else "no_online_learning", "family": family,
           "collection_only": online.updates == 0, "interactions": len(rows),
           "updates": primary_delta, "primary_optimizer": primary,
           "requested_primary_optimizer_updates": online.updates,
           "executed_primary_optimizer_updates": primary_delta,
           "executed_optimizer_steps_total": final.optimizer_steps - checkpoint.optimizer_steps,
           "transition_records": raw_rows, "complete_episodes": complete, "partial_episode": partial,
           "complete_episode_mean_return": float(np.mean([r["return"] for r in complete])) if complete else None,
           "episode_seeds": reset_seeds, "replay_reset_at_scenario_start": True,
           "pretrained_checkpoint_sha256": checkpoint.digest,
           "restored_checkpoint_sha256": checkpoint.digest,
           "configured_checkpoint_sha256": configured_checkpoint.digest,
           "end_checkpoint_sha256": final.digest,
           "learning_settings_requested": _settings(online.learning_settings),
           "learning_settings_applied": online.updates > 0,
           "learning_settings_before": _effective_settings(setting_before),
           "learning_settings_after": _effective_settings(final),
           "learning_settings_before_sha256": _hash(_effective_settings(setting_before)),
           "learning_settings_after_sha256": _hash(_effective_settings(final)),
           "optimizer_step_deltas": _counts_delta(checkpoint, final),
           "optimizer_step_counts": dict(final.optimizer_step_counts),
           "replay": {"rounding": "floor(batch_size * offline_fraction); remainder online",
                      "offline_per_batch": n_offline, "online_per_batch": n_online,
                      "sampled_offline": sampled_offline, "sampled_online": sampled_online,
                      "with_replacement": True, "offline_split": "train"},
           "last_update_metrics": metrics,
           "return_role": "adaptation diagnostics only; excluded from frozen evaluation results"}
    return final, raw


def run_mode(dataset, pretrained_checkpoint, factory, config=EvaluationConfig(), *, families=FAMILIES,
             checkpoint_directory=None):
    """Restore each scenario independently, optionally adapt, then freeze anew.

    Returned checkpoints are objects; all other artifacts are JSON-compatible.
    With ``checkpoint_directory``, exclusively save every endpoint, reload and
    verify each digest, and restore all learners before any frozen evaluation.
    Without a destination the receipt explicitly identifies in-memory snapshots.
    Other result artifacts may be persisted after the inference guard exits.
    A false flag with zero updates is labeled ``no_online_learning``. A positive
    interaction budget still collects transitions without changing settings.
    """
    assert_training_allowed("run_mode")
    if not isinstance(config, EvaluationConfig) or not isinstance(pretrained_checkpoint, LearnerCheckpoint):
        raise RunnerError("run_mode requires EvaluationConfig and LearnerCheckpoint")
    metadata = _dataset(dataset)
    families = _validate_families(families)
    identity = _identity(dataset)
    bound = pretrained_checkpoint.metadata.get("runner_state", {})
    if bound.get("dataset_sha256") != dataset.digest or bound.get("train_split_sha256") != identity["split_sha256"]["train"]:
        raise RunnerError("pretrained checkpoint is not bound to this dataset and train split")
    collect = (not config.frozen_policy and config.online is not None and config.online.interactions > 0)
    adapt = collect and config.online.updates > 0
    learners = []
    for _ in families:
        learner = factory()
        if any(learner is earlier for earlier in learners):
            raise RunnerError("factory must return a new learner instance for every scenario")
        _capabilities(learner, online=collect)
        _restore(learner, pretrained_checkpoint)
        learners.append(learner)
    checkpoint_directory = _checkpoint_destination(checkpoint_directory)
    checkpoints, adaptation, evaluations = {}, {}, {}
    offline_seeds = {int(seed) for seed in dataset.arrays["episode_seeds"]}
    forbidden = set(offline_seeds)
    for family, learner in zip(families, learners):
        if collect:
            final, raw = _adapt(dataset, learner, pretrained_checkpoint, config, family, forbidden)
        else:
            final = _checkpoint(learner)
            raw = {"status": "frozen_policy" if config.frozen_policy else "no_online_learning",
                   "family": family, "interactions": 0, "updates": 0,
                   "primary_optimizer": learner.capabilities.primary_optimizer,
                   "requested_primary_optimizer_updates": 0 if config.online is None else config.online.updates,
                   "collection_only": False, "executed_primary_optimizer_updates": 0,
                   "executed_optimizer_steps_total": 0,
                   "transition_records": [], "complete_episodes": [], "partial_episode": None,
                   "episode_seeds": [], "pretrained_checkpoint_sha256": pretrained_checkpoint.digest,
                   "restored_checkpoint_sha256": final.digest, "end_checkpoint_sha256": final.digest,
                   "learning_settings_applied": False, "optimizer_step_deltas": _counts_delta(final, final)}
        checkpoints[family], adaptation[family] = final, raw
    checkpoint_artifacts = {}
    if checkpoint_directory is not None:
        # A new directory and exclusive file writes refuse both pre-existing
        # artifacts and destinations created after the read-only preflight.
        checkpoint_directory.mkdir(parents=True, exist_ok=False)
        for family, checkpoint in checkpoints.items():
            path = checkpoint.save(checkpoint_directory / f"{family}.json")
            checkpoint_artifacts[family] = {"path": str(path), "sha256": checkpoint.digest}
        for family, learner in zip(families, learners):
            loaded = LearnerCheckpoint.load(checkpoint_artifacts[family]["path"])
            if loaded.digest != checkpoints[family].digest:
                raise RunnerError("saved checkpoint changed before frozen evaluation")
            _restore(learner, loaded)
            checkpoints[family] = loaded
    evaluation_checkpoint_source = ("reloaded_verified_checkpoint" if checkpoint_directory is not None
                                    else "in_memory_snapshot")
    # Evaluation seeds are independent of adaptation and excluded from every
    # observed reset. Equal seed lists also pair the evaluated scenario families.
    eval_rng = _rng(config.seed, "evaluation_resets")
    eval_seeds = [_fresh_seed(eval_rng, forbidden) for _ in range(config.episodes)]
    for family, learner in zip(families, learners):
        before = _checkpoint(learner)
        policy = learner.freeze()
        if _checkpoint(learner).digest != before.digest:
            raise RunnerError("freeze() mutated learner checkpoint state")
        evaluations[family] = evaluate_frozen(policy, family=family, episodes=config.episodes,
                  episode_seeds=eval_seeds, physics=metadata["physics"],
                  task_semantics=metadata["task_semantics"], provenance=metadata["provenance"])
        if _checkpoint(learner).digest != before.digest:
            raise RunnerError("frozen evaluation mutated the learner checkpoint state")
        evaluations[family]["checkpoint_sha256"] = before.digest
        evaluations[family]["evaluation_checkpoint_source"] = evaluation_checkpoint_source
        evaluations[family]["checkpoint_path"] = checkpoint_artifacts.get(family, {}).get("path")
    config_record = {**config.to_dict(), "families": list(families)}
    online_requested = None if config.online is None else config.online.to_dict()
    receipt = {"schema": "thrusterchain.training_mode.v4", **identity,
               "mode": "strict_offline" if config.frozen_policy else "offline_to_online",
               "config": config_record, "config_sha256": _hash(config_record),
               "status": "online_adaptation_then_frozen_evaluation" if adapt else (
                   "frozen_policy" if config.frozen_policy else "no_online_learning"),
               "collection_only": collect and not adapt,
               "learning_settings_applied": adapt,
               "primary_optimizer": learners[0].capabilities.primary_optimizer,
               "requested_primary_optimizer_updates_per_family": 0 if config.online is None else config.online.updates,
               "requested_primary_optimizer_updates": (0 if config.online is None else config.online.updates) * len(families),
               "pretrained_checkpoint_sha256": pretrained_checkpoint.digest,
               "checkpoint_persistence": "saved_and_reloaded" if checkpoint_directory is not None else "in_memory_snapshot",
               "checkpoint_artifacts": checkpoint_artifacts,
               "persisted_before_frozen_evaluation": checkpoint_directory is not None,
               "evaluation_checkpoint_source": evaluation_checkpoint_source,
               "budget": {"source": "explicit_online_budget" if config.online is not None else "no_online_budget",
                          "requested": online_requested,
                          "effective": online_requested if collect else {"interactions": 0, "updates": 0},
                          "executed_per_family": {family: {"interactions": adaptation[family]["interactions"],
                              "updates": adaptation[family]["updates"]} for family in families}},
               "frozen_evaluation_episodes_per_family": config.episodes,
               "frozen_evaluation_steps": config.episodes * 100 * len(families),
               "adaptation_interactions": sum(record["interactions"] for record in adaptation.values()),
               "adaptation_updates": sum(record["updates"] for record in adaptation.values()),
               "executed_primary_optimizer_updates": sum(record["updates"] for record in adaptation.values()),
               "executed_optimizer_steps_total": sum(record["executed_optimizer_steps_total"] for record in adaptation.values()),
               "checkpoint_sha256": {family: checkpoint.digest for family, checkpoint in checkpoints.items()},
               "optimizer_step_counts": {family: dict(checkpoint.optimizer_step_counts)
                                         for family, checkpoint in checkpoints.items()},
               "random_streams": {"bit_generator": "PCG64", "named_stream_keys": STREAMS,
                   "sampling_and_reset_streams_independent": True,
                   "learner_rng": "restored from identical pretrained checkpoint independently for each scenario",
                   "evaluation_resets": "fresh; excluded from offline and adaptation reset seeds",
                   "evaluation_seeds_paired_across_families": True},
               "resume_scope": "exact offline continuation; online end checkpoints preserve full learner state, not replay resumption",
               "guard_scope": "cooperative instrumentation and declared state_bytes checks; no OS security sandbox",
               "adaptation_returns_in_frozen_results": False}
    receipt["receipt_sha256"] = _hash(receipt)
    return {"checkpoints": checkpoints, "raw_adaptation": adaptation,
            "frozen_evaluation": evaluations, "receipt": receipt}


__all__ = ["RunnerError", "train_transitions", "train_offline", "evaluate_frozen", "run_mode", "FAMILIES"]

"""One reserved, bounded final-policy evaluation; import executes no scientific work.

Uses the pinned canonical reset law/dynamics, float32 action deployment, raw
time-aware observations, and unchanged native actor normalization. Synthetic
tests exercise the accounting and mathematics without importing JAX or the core.
"""
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "bca_one_sided_v1"
PAYLOAD = HERE.parents[1] / "titan_20261026/payload"
BUNDLE = PAYLOAD / "transfer_bundle"
BUNDLE_SHA = "3443280fd4340ba64aef4d9601136257014bf2a6c61ca8c890eba06afa76690a"
MATRIX_SHA = "289afb581b4fdbc97fd68566c0b42ff6244b20686e9e49ba177e7e6bd57454a1"
CANONICAL_SHA = "f22f2e214ad9231bee4ce60ef55e64409534c2654c4e184406553ba8ed67dde1"
RUN_ORDER = ["seed0_native", "seed0_bca", "seed0_distance", "seed1_distance", "seed1_bca", "seed1_native"]
SEEDS = list(range(2026102800, 2026102832))
ADDITIONAL = {"critic": 2000, "actor": 1000, "actor_target": 1000,
              "critic_target": 1000, "outer_steps": 1000, "auxiliary": 2000,
              "aux_fit_attempts": 2000, "refresh_attempts": 1, "refresh_accepted": 1}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    return {"path": path.name, "sha256": sha(path.read_bytes()), "bytes": path.stat().st_size}


def _module(name, path):
    path = Path(path).resolve()
    if name in sys.modules:
        if Path(sys.modules[name].__file__).resolve() != path:
            raise ValueError("Conflicting helper import: " + name)
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def _checked(base, relative):
    base = Path(base).resolve()
    rel = Path(relative)
    path = base / rel
    if (rel.is_absolute() or ".." in rel.parts or ":" in str(relative)
            or path.is_symlink() or not path.resolve().is_relative_to(base)
            or not path.is_file()):
        raise ValueError("Unsafe or missing artifact: " + str(relative))
    return path.read_bytes()


def validate_protocol(protocol):
    expected = {"schema": "unifloral.matched_policy_pilot.v1", "gamma": .98,
        "horizon": 100, "evaluation_seeds": SEEDS, "evaluation_policies": 7,
        "evaluation_episodes": 224, "simulator_transition_cap": 22400,
        "evaluation_timeout_seconds": 600, "evaluation_bootstrap_draws": 2000,
        "evaluation_bootstrap_seed": 2026102691, "run_order": RUN_ORDER,
        "return_band_atol": 1e-6, "return_band_rtol": 1e-6,
        "critic_updates": 2000, "actor_updates": 1000, "auxiliary_updates": 2000,
        "absolute_critic_start": 20000, "absolute_critic_end": 22000,
        "absolute_refresh": 21000,
        "checkpoint_state_sha256": "874d65b7f4dcb5ec285a0d4360439c73a787b5e8c567ff7c0138e0c1e22948e8",
        "native_source_sha256": "6f5ab6f98d706d902a18f11e1c81e99e2691dc4c8a53843bf6521c9f95d00eee",
        "native_revision": "f2dc1278eae18ed3e22c92255119c54885414d17",
        "dataset_sha256": "4c857aaa0373bee4c74db33036303b4cf75c703b289a459525f75090741f3226"}
    for key, value in expected.items():
        if type(protocol.get(key)) is not type(value) or protocol[key] != value:
            raise ValueError("Evaluation protocol differs: " + key)


def observation(states, step, horizon):
    states = np.asarray(states)
    if states.ndim != 2 or states.shape[1] != 13 or states.dtype != np.float64:
        raise ValueError("Physical states must be float64 [batch,13]")
    if not 0 <= step <= horizon or not np.all(np.isfinite(states)):
        raise ValueError("Invalid physical state or time")
    return np.concatenate((states, np.full((len(states), 1), (horizon-step)/horizon)), axis=1).astype(np.float32)


def deployed_actions(actor, observations):
    raw = np.asarray(actor(observations))
    if raw.shape != (len(observations), 4) or raw.dtype.kind != "f" or not np.all(np.isfinite(raw)):
        raise ValueError("Actor must return finite [batch,4] actions")
    actions = raw.astype(np.float32)
    if not np.all(np.isfinite(actions)) or np.any(np.abs(actions) > 1):
        raise ValueError("Invalid native tanh action at float32 deployment boundary")
    return actions


class Ledger:
    """Persist charges before dispatch; uncertain work is never treated as zero."""
    def __init__(self, path, cap, timeout, clock=time.monotonic, started=None):
        self.path, self.clock = Path(path), clock
        self.started = clock() if started is None else started
        self.deadline, self.cap = self.started + timeout, cap
        self.data = {"status": "preparing_evaluation", "attempted_transition_rows": 0,
            "completed_transition_rows": 0, "inflight_transition_rows": 0,
            "execution_counts_known": True, "completed_episodes": 0,
            "completed_policies": [], "reset_calls_attempted": 0,
            "reset_calls_completed": 0, "reset_inflight": False,
            "actor_action_rows": 0, "logged_probe_action_rows": 0,
            "failed_policies": 0, "failure_count": 0,
            "optimizer_updates": 0, "calibration_updates": 0}
        self.persist()

    def persist(self):
        self.data["elapsed_seconds"] = self.clock()-self.started
        atomic_json(self.path, self.data)

    def deadline_check(self):
        if self.clock() >= self.deadline:
            raise TimeoutError("Evaluation wall-time limit reached; no extension")

    def before(self, rows, policy, step):
        self.deadline_check()
        if self.data["inflight_transition_rows"] or self.data["attempted_transition_rows"] + rows > self.cap:
            raise RuntimeError("Transition cap or unresolved prior dispatch")
        self.data.update(status="evaluating", policy=policy, step=step,
            attempted_transition_rows=self.data["attempted_transition_rows"]+rows,
            inflight_transition_rows=rows, execution_counts_known=False)
        self.persist()

    def after(self, rows):
        if self.data["inflight_transition_rows"] != rows:
            raise RuntimeError("Transition completion accounting mismatch")
        self.data.update(completed_transition_rows=self.data["completed_transition_rows"]+rows,
                         inflight_transition_rows=0, execution_counts_known=True)
        self.persist()

    def before_reset(self):
        self.deadline_check()
        if self.data["reset_inflight"] or self.data["inflight_transition_rows"]:
            raise RuntimeError("Unresolved prior reset or transition")
        self.data.update(reset_calls_attempted=self.data["reset_calls_attempted"]+1,
                         reset_inflight=True, execution_counts_known=False)
        self.persist()

    def after_reset(self):
        if not self.data["reset_inflight"]:
            raise RuntimeError("Reset completion accounting mismatch")
        self.data.update(reset_calls_completed=self.data["reset_calls_completed"]+1,
                         reset_inflight=False, execution_counts_known=True)
        self.persist()


def rollout(actor, dynamics, starts, horizon, gamma, ledger, policy, check_state=None):
    """Canonical dynamics callback; all policies choose their own actions each step."""
    states = np.asarray(starts, dtype=np.float64).copy()
    n = len(states)
    trajectory = np.empty((n, horizon+1, 13), np.float64)
    actions = np.empty((n, horizon, 4), np.float32)
    rewards = np.empty((n, horizon), np.float64)
    returns = np.zeros(n, np.float64)
    trajectory[:, 0] = states
    for step in range(horizon):
        ledger.deadline_check()
        action = deployed_actions(actor, observation(states, step, horizon))
        ledger.data["actor_action_rows"] += n
        ledger.before(n, policy, step)
        nxt, reward = dynamics(states, action.astype(np.float64))
        # A returned callback establishes the number of completed transitions,
        # even if its numeric output subsequently fails validation.
        ledger.after(n)
        nxt, reward = np.asarray(nxt), np.asarray(reward)
        if (nxt.shape != (n, 13) or nxt.dtype != np.float64
                or reward.shape != (n,) or reward.dtype != np.float64
                or not np.all(np.isfinite(nxt)) or not np.all(np.isfinite(reward))):
            raise ValueError("Nonfinite or malformed canonical transition")
        if check_state is not None:
            check_state(nxt, step+1)
        trajectory[:, step+1], actions[:, step], rewards[:, step] = nxt, action, reward
        # Explicit left-to-right summation, including reward at t=99; no bootstrap.
        returns += (gamma ** step) * reward
        states = nxt
        ledger.deadline_check()
    if not np.all(np.isfinite(returns)):
        raise ValueError("Nonfinite discounted episode return")
    ledger.data["completed_episodes"] += n
    ledger.data["completed_policies"].append(policy)
    ledger.persist()
    return {"states": trajectory, "actions": actions, "rewards": rewards, "returns": returns}


def paired_comparison(left, right, protocol, indices):
    left, right = np.asarray(left, np.float64), np.asarray(right, np.float64)
    if left.shape != (32,) or right.shape != (32,) or not np.all(np.isfinite([left, right])):
        raise ValueError("Paired comparison requires 32 finite returns")
    difference = left-right
    band = protocol["return_band_atol"] + protocol["return_band_rtol"] * np.maximum(np.abs(left), np.abs(right))
    means = difference[indices].mean(axis=1)
    interval = np.quantile(means, [.025, .975], method="linear")
    return {"paired_differences": difference.tolist(), "mean_gain": float(difference.mean()),
        "mean_numerical_band": float(band.mean()), "positive_screen": bool(difference.mean() > band.mean()),
        "improved_episodes": int(np.sum(difference > band)),
        "harmed_episodes": int(np.sum(difference < -band)),
        "numerical_ties": int(np.sum(np.abs(difference) <= band)),
        "harm_episode_indices": np.flatnonzero(difference < -band).tolist(),
        "mean_gain_episode_paired_bootstrap_95pct": interval.tolist()}


def decision_report(returns, protocol):
    rng = np.random.Generator(np.random.PCG64(protocol["evaluation_bootstrap_seed"]))
    indices = rng.integers(0, 32, size=(protocol["evaluation_bootstrap_draws"], 32))
    comparisons = {}
    for stream in (0, 1):
        comparisons[str(stream)] = {control: paired_comparison(
            returns[f"seed{stream}_bca"], returns["original" if control == "original" else f"seed{stream}_{control}"],
            protocol, indices) for control in ("native", "distance", "original")}
    passed = all(comparisons[str(s)][c]["positive_screen"] for s in (0, 1) for c in ("native", "distance"))
    return {"decision": "promising_exploratory_screen" if passed else "stop_this_pilot_without_tuning",
        "primary_screen_passed": passed, "comparisons_by_continuation_stream": comparisons,
        "interval_scope": "Descriptive paired-episode percentile intervals conditional on each fixed final actor. Two continuation streams share one trained initialization; no independent-trained-seed or superiority inference.",
        "bootstrap": {"draws": protocol["evaluation_bootstrap_draws"], "seed": protocol["evaluation_bootstrap_seed"],
            "generator": "NumPy PCG64", "shared_paired_indices_across_comparisons": True,
            "indices_sha256": sha(indices.astype(np.int64).tobytes(order="C"))}}


def displacement(actions, anchor_actions, states, anchor_states, original_std):
    action_mse = np.mean((np.asarray(actions, np.float64)-np.asarray(anchor_actions, np.float64))**2, axis=1)
    scale = np.asarray(original_std, np.float64)[:13] + .001
    standardized = (np.asarray(states, np.float64)[:, 1:]-np.asarray(anchor_states, np.float64)[:, 1:]) / scale
    state_rms = np.sqrt(np.mean(standardized**2, axis=2))
    if action_mse.shape != (128,) or state_rms.shape != (32, 100) or not np.all(np.isfinite(state_rms)):
        raise ValueError("Shift probe or trajectory shape differs")
    return {"logged_probe_action_mse": float(action_mse.mean()), "logged_probe_action_mse_by_row": action_mse.tolist(),
        "mean_standardized_physical_state_rms": float(state_rms.mean()),
        "standardized_physical_state_rms_by_episode": state_rms.mean(axis=1).tolist(),
        "state_definition": "All13 physical coordinates, steps1..100, divided by unchanged original training std[:13]+0.001; RMS across13 then equal time/episode mean. Remaining-time coordinate excluded. Original mean cancels in paired differences.",
        "scope": "Descriptive action and induced state change relative to original actor, not an OOD validity or efficacy gate."}


def leaf_descriptions(value, path=""):
    if isinstance(value, dict):
        return [item for key in sorted(value) for item in leaf_descriptions(value[key], path+"/"+str(key) if path else str(key))]
    if isinstance(value, (tuple, list)):
        return [item for i, child in enumerate(value) for item in leaf_descriptions(child, path+"/"+str(i) if path else str(i))]
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or not np.all(np.isfinite(array)):
        raise ValueError("Invalid actor parameter leaf")
    return [{"path": path, "shape": list(array.shape), "dtype": array.dtype.str, "sha256": sha(array.tobytes(order="C"))}]


def validate_actor_manifest(manifest, run_id, protocol, protocol_sha, normalization, initial_counts):
    stream, arm = int(run_id[4]), run_id[6:]
    expected = {"schema": "unifloral.matched_policy.actor.v1", "run_id": run_id,
        "stream_index": stream, "arm": arm, "completed": True, "phase": 0,
        "protocol_sha256": protocol_sha, "native_revision": protocol["native_revision"],
        "native_source_sha256": protocol["native_source_sha256"],
        "initial_checkpoint_state_sha256": protocol["checkpoint_state_sha256"],
        "additional_updates": ADDITIONAL,
        "architecture": {"class": "DeterministicTanhActor", "source": "pinned standalone UniFloral TD3+BC", "action_dim": 4, "observation_dim": 14}}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError("Actor manifest mismatch: " + run_id + "/" + key)
    norm = manifest.get("normalization", {})
    if norm.get("epsilon") != .001 or any(norm.get(k) != normalization[k] for k in ("mean", "std")):
        raise ValueError("Actor normalization changed")
    start, end = manifest.get("initial_absolute_counts", {}), manifest.get("final_absolute_counts", {})
    # Manifest initial counts identify the original state before the fork.
    for key, wanted in initial_counts.items():
        if start.get(key) != wanted:
            raise ValueError("Initial training counters differ: " + key)
    for key, change in ADDITIONAL.items():
        if end.get(key) != start[key] + change:
            raise ValueError("Final training counter differs: " + key)
    if (end.get("phase") != 0 or end.get("critic_budget") != 22000
            or end.get("last_refresh_critic") != 21000 or end.get("reference_ready") is not True
            or end.get("next_refresh_index") != start["next_refresh_index"]+1):
        raise ValueError("Final schedule/reference counters differ")


def training_seal(directory, run_id, protocol_sha):
    """Bind successful owned-child receipt to report and exact actor manifest."""
    supervisor_raw = _checked(directory, "SUPERVISOR_RESULT.json")
    report_raw = _checked(directory, "report.json")
    manifest_raw = _checked(directory, "actor_manifest.json")
    supervisor, report = json.loads(supervisor_raw), json.loads(report_raw)
    if (supervisor.get("job") != run_id or supervisor.get("completed") is not True
            or supervisor.get("exit_code") != 0 or supervisor.get("timed_out") is not False
            or supervisor.get("interrupted") is not False
            or supervisor.get("report_sha256") != sha(report_raw)
            or report.get("status") != "completed_bounded_policy_training"
            or report.get("run_id") != run_id or report.get("protocol_sha256") != protocol_sha
            or report.get("execution_counts_known") is not True
            or report.get("actor_manifest_sha256") != sha(manifest_raw)):
        raise ValueError("Training supervisor/report/manifest seal differs: " + run_id)
    return manifest_raw, {"SUPERVISOR_RESULT.json": sha(supervisor_raw),
                          "report.json": sha(report_raw), "actor_manifest.json": sha(manifest_raw)}


def core_identity():
    raw = _checked(BUNDLE, "BUNDLE_MANIFEST.json")
    if sha(raw) != BUNDLE_SHA:
        raise ValueError("Canonical bundle manifest changed")
    files = {k: v for k, v in json.loads(raw)["files"].items() if k.startswith("core_v4/")}
    if not files:
        raise ValueError("Canonical source map absent")
    for path, wanted in files.items():
        if sha(_checked(BUNDLE, path)) != wanted:
            raise ValueError("Canonical source changed: " + path)
    return {"bundle_manifest_sha256": BUNDLE_SHA, "verified_core_files": len(files),
            "canonical_source_sha256": CANONICAL_SHA, "matrix_sha256": MATRIX_SHA}


def load_core():
    identity = core_identity()
    path = (BUNDLE / "core_v4").resolve()
    sys.path.insert(0, str(path))
    core = importlib.import_module("thrusterchain.environment")
    if Path(core.__file__).resolve() != path / "thrusterchain/environment.py":
        raise ValueError("Different ThrusterChain core already imported")
    if core.dynamics is not core.canonical.dynamics:
        raise ValueError("Canonical dynamics binding differs")
    config = core.canonical_config()
    expected = {"physics": {"stall": True, "stall_slope": 1.5, "knee": 0.800000011920929,
        "gain": .3, "throttle_cost": .3, "wave_turn": .27, "gamma": .98, "length": 100},
        "state_dim": 13, "action_dim": 4, "observation_dim": 14,
        "observation_mode": "time_aware", "task_semantics": "finite_horizon",
        "objective_gamma": .98, "horizon": 100, "terminal_reward": 0., "terminal_bootstrap": False,
        "matrix_sha256": MATRIX_SHA}
    if config != expected:
        raise ValueError("Canonical task/config differs")
    return core, {**identity, "module": str(Path(core.__file__).resolve()), "config": config}


def run(protocol, output_dir, training_outputs):
    """Called only by the reserved launcher child after all six final exports exist."""
    started = time.monotonic()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Separate guard ensures a repeated evaluator dispatch cannot duplicate rollout.
    with (output / "EVALUATION_ATTEMPT.json").open("xb") as stream:
        stream.write(encoded({"schema": "unifloral.matched_policy.evaluation_attempt.v1",
            "automatic_retry": False, "run_order": ["original"]+RUN_ORDER}))
    ledger = Ledger(output / "progress.json", 22400, 600, started=started)
    try:
        validate_protocol(protocol)
        protocol_raw = (HERE / "protocol.json").read_bytes()
        if json.loads(protocol_raw) != protocol:
            raise ValueError("Supplied protocol differs from sealed protocol file")
        protocol_sha = sha(protocol_raw)
        seed_raw = (HERE / "EVALUATION_SEEDS.json").read_bytes()
        seed_manifest = json.loads(seed_raw)
        if (seed_manifest["reset_seeds"] != SEEDS or seed_manifest["known_reset_or_reserved_confirmation_collisions"]
                or seed_manifest["dataset_sha256"] != protocol["dataset_sha256"]):
            raise ValueError("Reset seed manifest mismatch/collision")
        if set(training_outputs) != set(RUN_ORDER):
            raise ValueError("All six final trained policies are required")
        restore = _module("_matched_eval_restore", PRIOR / "restore_logged.py")
        operator_protocol = json.loads((restore.ROOT / "protocol.json").read_bytes())
        restored = restore.load(operator_protocol)
        import jax
        import jax.numpy as jnp
        from flax import serialization
        if jax.default_backend() != "gpu":
            raise ValueError("Approved GPU runtime required")
        actor = restored.state.microstep.agent_state.actor
        normalization = {"mean": np.asarray(restored.context.obs_mean).tolist(), "std": np.asarray(restored.context.obs_std).tolist()}
        original_description = leaf_descriptions(serialization.to_state_dict(actor.params))
        template_structure = [{k: row[k] for k in ("path", "shape", "dtype")} for row in original_description]
        params, actor_identities, input_hashes = {"original": actor.params}, {}, {}
        for run_id in RUN_ORDER:
            directory = Path(training_outputs[run_id]).resolve()
            raw, sealed_hashes = training_seal(directory, run_id, protocol_sha)
            manifest = json.loads(raw)
            validate_actor_manifest(manifest, run_id, protocol, protocol_sha, normalization, restore.COUNTS)
            for key, filename in (("actor_params", "final_actor.msgpack"), ("final_state", "final_state.msgpack")):
                entry = manifest[key]
                data = _checked(directory, filename)
                if entry != {"path": filename, "sha256": sha(data), "bytes": len(data)}:
                    raise ValueError("Export file digest differs: " + run_id + "/" + filename)
                input_hashes[str(directory / filename)] = sha(data)
                if key == "actor_params":
                    params[run_id] = serialization.from_bytes(actor.params, data)
            desc = leaf_descriptions(serialization.to_state_dict(params[run_id]))
            if desc != manifest["parameter_leaves"] or [{k: row[k] for k in ("path", "shape", "dtype")} for row in desc] != template_structure:
                raise ValueError("Actor numeric leaves/template mismatch: " + run_id)
            input_hashes.update({str(directory / name): digest for name, digest in sealed_hashes.items()})
            actor_identities[run_id] = {"directory": str(directory), "manifest_sha256": sha(raw), "actor_params": manifest["actor_params"],
                "sealed_hashes": sealed_hashes, "initial_absolute_counts": manifest["initial_absolute_counts"], "final_absolute_counts": manifest["final_absolute_counts"]}
        params_before = {key: leaf_descriptions(serialization.to_state_dict(value)) for key, value in params.items()}
        apply = jax.jit(lambda p, obs: actor.apply_fn(p, obs).sample(seed=None))
        actors = {key: (lambda observations, p=value: np.asarray(apply(p, jnp.asarray(observations, jnp.float32)))) for key, value in params.items()}
        # Only the fixed logged validation probe is used; no outcomes enter scores.
        logged = _module("_matched_eval_logged", PRIOR / "logged_data.py")
        banks = logged.load_logged_banks()
        probe = banks["validation"]
        if probe["obs"].shape != (128, 14):
            raise ValueError("Fixed logged probe changed")
        core, core_receipt = load_core()
        physics = core.Physics(stall=True, stall_slope=1.5)
        starts = []
        for seed in SEEDS:
            ledger.before_reset()
            env = core.ThrusterChainEnv(stall=True, start="recorded", stall_slope=1.5,
                observation_mode="time_aware", task_semantics="finite_horizon")
            try:
                obs, info = env.reset(seed=seed)
                state = env.state
                ledger.after_reset()
                if (env.t != 0 or info["remaining_steps"] != 100 or info["bootstrap_mask"] != 1.
                        or info["continuation_discount"] != .98
                        or not np.array_equal(obs, observation(state[None], 0, 100)[0])):
                    raise ValueError("Canonical reset/observation semantics differ")
                starts.append(state)
            finally:
                env.close()
        starts = np.stack(starts)
        starts_receipt = atomic_npz(output / "reset_states.npz", reset_seeds=np.asarray(SEEDS, np.int64), states=starts)
        def check_state(states, step):
            if np.any(core.canonical.float_radius(states) > core.canonical.FLOAT_BOUND):
                raise ValueError("Canonical float bound exceeded")
            obs = observation(states, step, 100)
            low = np.r_[core.canonical.OBS_LOW, np.float32(0)]
            high = np.r_[core.canonical.OBS_HIGH, np.float32(1)]
            if np.any(obs < low) or np.any(obs > high):
                raise ValueError("Canonical observation space exceeded")
        returns, shifts, files = {}, {}, {}
        original_trajectory = original_probe_actions = None
        for policy in ["original"]+RUN_ORDER:
            result = rollout(actors[policy], lambda s, a: core.dynamics(s, a, physics), starts,
                             100, .98, ledger, policy, check_state)
            files[policy] = atomic_npz(output / (policy + ".npz"), **result, reset_seeds=np.asarray(SEEDS, np.int64))
            ledger.deadline_check()
            probe_actions = deployed_actions(actors[policy], probe["obs"])
            ledger.data["logged_probe_action_rows"] += 128
            ledger.persist()
            if policy == "original":
                original_trajectory, original_probe_actions = result["states"], probe_actions
            returns[policy] = result["returns"]
            shifts[policy] = displacement(probe_actions, original_probe_actions, result["states"], original_trajectory, normalization["std"])
        ledger.deadline_check()
        immutable = restored.assert_unchanged()
        for key, value in params.items():
            if leaf_descriptions(serialization.to_state_dict(value)) != params_before[key]:
                raise ValueError("Evaluation mutated actor parameters: " + key)
        for path, digest in input_hashes.items():
            if sha(Path(path).read_bytes()) != digest:
                raise ValueError("Training export changed during evaluation")
        if (sha((HERE / "protocol.json").read_bytes()) != protocol_sha
                or sha((HERE / "EVALUATION_SEEDS.json").read_bytes()) != sha(seed_raw)
                or core_identity() != {k: core_receipt[k] for k in ("bundle_manifest_sha256", "verified_core_files", "canonical_source_sha256", "matrix_sha256")}):
            raise ValueError("Evaluation inputs/source changed during execution")
        if ledger.data["completed_transition_rows"] != 22400 or ledger.data["completed_episodes"] != 224:
            raise ValueError("Final evaluation budget accounting differs")
        report = {"status": "completed_bounded_policy_evaluation", **decision_report(returns, protocol),
            "counts": ledger.data.copy(), "protocol_sha256": protocol_sha,
            "evaluation_seed_manifest_sha256": sha(seed_raw), "reset_seed_scope": seed_manifest["scope"],
            "core": core_receipt, "checkpoint_restore": restored.receipt, "actor_identities": actor_identities,
            "normalization": {**normalization, "epsilon": .001, "changed": False},
            "reset_states": starts_receipt, "policy_results": files,
            "returns_by_policy": {key: value.tolist() for key, value in returns.items()},
            "mean_discounted_returns": {key: float(value.mean()) for key, value in returns.items()},
            "action_and_state_displacement": shifts,
            "logged_probe": {"rows": 128, "row_ids": probe["row_ids"].tolist(), "group_ids": probe["group_ids"].tolist(),
                "observation_sha256": sha(probe["obs"].tobytes(order="C")), "role": "Existing logged development validation, no fitting or calibration updates"},
            "immutability": immutable, "actor_parameters_unchanged": True,
            "failure_count": 0, "execution_counts_known": True,
            "limitations": ["Exploratory nominal-task pilot following development and reused calibration; no confirmation claim.",
                "Two continuation streams share one trained initialization; episode intervals do not estimate independent-training variation.",
                "No OOD environment shift, candidate reranking, evaluation-driven updates, intermediate selection, or automatic extension.",
                "Zeus runtime remains outside upstream JAX/NumPy package constraints as previously disclosed."]}
        ledger.deadline_check()
        ledger.data["status"] = report["status"]
        ledger.persist()
        report["counts"] = ledger.data.copy()
        atomic_json(output / "report.json", report)
        return report
    except BaseException as error:
        ledger.data.update(status="failed_bounded_policy_evaluation", failure_count=1,
            failed_policies=int(ledger.data.get("status") == "evaluating"),
            error=type(error).__name__ + ": " + str(error))
        ledger.persist()
        report = {"status": ledger.data["status"], "error": ledger.data["error"],
            "counts": ledger.data.copy(), "execution_counts_known": ledger.data["execution_counts_known"],
            "automatic_retry": False, "decision": "incomplete_no_utility_conclusion"}
        atomic_json(output / "report.json", report)
        raise

"""Reproducible light checks: python -m experiments.thrusterchain_diagnostics smoke --output NEW_DIR."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy

from .finite import (FiniteConfig, FiniteMDP, solve_optimal, evaluate_policy,
                     occupancies, support_ratio, cap_policy, hidden_kernel_certificate)
from .data import (RetentionSpec, behavior_policy, generate_trajectory_data,
                   generate_fixed_state_data, generate_evaluation_data, paired_hidden_control)
from .diagnostics import MethodOutput, evaluate_method
from .continuous import (Objective, FiniteHorizonQP, high_load_start,
                         fixed_cap_policy, regime_cap_policy, policy_value, provenance)
from .continuous_data import generate_continuous_data

HERE = Path(__file__).resolve().parent


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")


def _save_dataset(root, name, data):
    np.savez_compressed(root / "learner" / (name + ".npz"), **data.learner_rows)
    np.savez_compressed(root / "evaluator" / (name + "_audit.npz"), **data.evaluator)
    _json(root / "learner" / (name + ".json"), data.metadata)
    rows = data.evaluator
    return dict(generated_rows=len(rows["reward"]), retained_rows=int(rows["retained"].sum()),
                critical_generated=int(rows["critical"].sum()),
                critical_retained=int((rows["critical"] & rows["retained"]).sum()))


def run_smoke(output, config):
    """No learner fitting, checkpoint selection, or overwriting previous runs."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"output already exists; choose a new directory: {output}")
    if config.get("optimization_updates") != 0 or config.get("online_selection_steps") != 0 or config.get("budget_topups") != 0:
        raise ValueError("this entry point only supports zero-training, zero-selection, zero-topup smoke checks")
    start = time.perf_counter()
    output.mkdir(parents=True)
    (output / "learner").mkdir()
    (output / "evaluator").mkdir()
    _json(output / "config.json", config)
    report = dict(schema="thrusterchain-diagnostics-report-v1", status="smoke_only_no_learned_methods",
                  runtime=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                  budgets=dict(optimization_updates=0, online_selection_steps=0, budget_topups=0),
                  continuous={}, finite={}, data={})
    mdp = FiniteMDP(FiniteConfig(**config["finite"]))
    optimal = solve_optimal(mdp)
    beta = behavior_policy(mdp)
    initial = mdp.initial_distribution("overlap")
    logging_occ = occupancies(mdp, beta, initial).state_action
    n = config["trajectory_episodes"]
    seed = config["seeds"]["data"]
    retained_counts = None
    for level in ("full", "rare", "zero"):
        retention = RetentionSpec(level, config["rare_retention_probability"])
        data = generate_trajectory_data(mdp, n, seed=seed, retention=retention)
        report["data"]["finite_trajectory_" + level] = _save_dataset(output, "finite_trajectory_" + level, data)
        fixed = generate_fixed_state_data(mdp, config["fixed_samples_per_state_action"], seed=seed, retention=retention)
        report["data"]["finite_fixed_" + level] = _save_dataset(output, "finite_fixed_" + level, fixed)
        if level == "zero":
            retained_counts = np.zeros_like(logging_occ, dtype=int)
            rows = data.learner_rows
            np.add.at(retained_counts, (rows["time"], rows["state"], rows["action"]), 1)
    holdout = generate_evaluation_data(mdp, n, seed=config["seeds"]["holdout"])
    # Holdout is evaluator-only, never exported into the learner directory.
    np.savez_compressed(output / "evaluator" / "finite_holdout.npz", **holdout.evaluator)
    _json(output / "evaluator" / "finite_holdout.json", holdout.metadata)
    raw = np.zeros_like(beta)
    action_idx = np.argmin(np.abs(mdp.actions[None, :] + np.sign(mdp.state_position)[:, None]), axis=1)
    raw[:, np.arange(mdp.n_states), action_idx] = 1
    raw[-1] = 0; raw[-1, :, 2] = 1  # finite terminal state is uncharged
    fixed_cap = cap_policy(mdp, raw)
    aware_cap = cap_policy(mdp, raw, regime_aware=True)
    logging_mass = logging_occ * RetentionSpec("zero").propensities(mdp.critical_mask())[None]
    control_policies = {"raw_feedback": raw, "fixed_cap": fixed_cap, "regime_cap": aware_cap,
                        "optimal_evaluator_control": optimal.policy}
    controls = {}
    for name, policy in control_policies.items():
        controls[name] = evaluate_method(mdp, MethodOutput(policy), baseline_policy=fixed_cap,
                                        logging_mass=logging_mass, retained_counts=retained_counts,
                                        critical_mask=mdp.critical_mask(), failure_mask=np.abs(mdp.state_position) >= 2)
    report["finite"]["controls"] = controls
    report["finite"]["contract"] = dict(states=mdp.n_states, actions=mdp.n_actions,
                                        config=asdict(mdp.config), exactness="finite model enumeration, floating-point arithmetic")
    witness_position = int(np.argmin(np.abs(mdp.positions - 1.5)))
    report["finite"]["high_action_witness"] = {
        name: float(optimal.Q[0, mdp.state_index(witness_position, load=1, regime=r), 0]
                    - optimal.Q[0, mdp.state_index(witness_position, load=1, regime=r), 1])
        for r, name in enumerate(("stall_q_high_minus_knee", "monotone_q_high_minus_knee"))}
    target_occ = occupancies(mdp, optimal.policy, initial).state_action
    shifts = {}
    for law in ("overlap", "thin", "zero"):
        source_initial = mdp.initial_distribution(law)
        context_ratio = support_ratio(initial, source_initial)
        source_occ = occupancies(mdp, beta, source_initial).state_action
        occupancy_ratio = support_ratio(target_occ, source_occ)
        shifts[law] = dict(context_unsupported_probability=context_ratio.target_mass_unsupported,
                           max_supported_context_ratio=float(np.nanmax(context_ratio.ratio)),
                           policy_unsupported_mass_per_step=occupancy_ratio.target_mass_unsupported / mdp.horizon)
    report["finite"]["context_vs_policy_shifts"] = shifts
    # Deliberately synthetic adapter checks: not trained predictions or evidence of calibration.
    dp = evaluate_policy(mdp, raw)
    intervals = np.stack((dp.Q - .1, dp.Q + .1), axis=-1)
    report["finite"]["synthetic_adapter_check"] = evaluate_method(
        mdp, MethodOutput(raw, q_values=dp.Q, q_intervals=intervals, refusal=np.ones(raw.shape[:2])),
        fallback_policy=fixed_cap, baseline_policy=fixed_cap)
    pair = paired_hidden_control(mdp.config, n_episodes=n, seed=seed)
    # Export exactly one shared learner log. World-specific probe material stays privileged.
    report["data"]["hidden_common_log"] = _save_dataset(output, "hidden_common_log", pair.stall)
    for world, probe in pair.probes.items():
        np.savez_compressed(output / "evaluator" / ("hidden_probe_" + world + ".npz"), **probe)
    report["finite"]["hidden_pair"] = dict(kernel_certificate=hidden_kernel_certificate(mdp.config),
                                             coupled_log_checks=pair.checks)
    from .data import rare_probe_control
    rare = rare_probe_control(mdp.config, n_episodes=n, seed=seed,
                              probe_probability=config["rare_probe_probability"])
    report["finite"]["rare_probe"] = rare.report
    # Rare-probe data are separate from the zero-information learner log.
    for world, dataset in (("stall", rare.stall), ("monotone", rare.monotone)):
        np.savez_compressed(output / "evaluator" / ("rare_probe_" + world + ".npz"), **dataset.evaluator)
    objective = Objective(**config["continuous"])
    x0 = high_load_start()
    qp = FiniteHorizonQP(objective)
    for regime, name in enumerate(("stall", "monotone")):
        solution = qp.solve(x0, regime)
        report["continuous"][name] = solution.as_dict()
        report["continuous"][name]["fixed_cap_return"] = policy_value(x0, regime, fixed_cap_policy(objective=objective), objective)
        report["continuous"][name]["regime_cap_return"] = policy_value(x0, regime, regime_cap_policy(objective), objective)
        np.savez_compressed(output / "evaluator" / ("continuous_qp_" + name + ".npz"), actions=solution.actions, start=x0)
    report["continuous"]["contract"] = dict(objective=objective.as_dict(), start="deterministic high_load_start, current1.5",
                                               population_expectation=False, provenance=provenance())
    for track in ("trajectory", "fixed_state"):
        for level in ("full", "rare", "zero"):
            data = generate_continuous_data(track=track, n_episodes=config["continuous_episodes"], objective=objective,
                                            seed=seed, retention=level, rare_probability=config["rare_retention_probability"])
            key = "continuous_" + track + "_" + level
            report["data"][key] = _save_dataset(output, key, data)
    arrays = dict(P=mdp.P, R=mdp.R, optimal_V=optimal.V, optimal_Q=optimal.Q,
                  logging_occupancy=logging_occ, optimal_occupancy=target_occ)
    np.savez_compressed(output / "evaluator" / "finite_oracle.npz", **arrays)
    # Only computational inputs; the independently regenerated HTML builder is
    # not imported by this run and may embed the resulting report afterwards.
    computational_files = ("__init__.py", "__main__.py", "finite.py", "continuous.py",
                           "data.py", "continuous_data.py", "diagnostics.py")
    report["source_sha256"] = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                               for name in computational_files}
    report["elapsed_seconds"] = time.perf_counter() - start
    _json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["smoke"])
    parser.add_argument("--output", required=True, type=Path, help="New directory; existing output is refused")
    parser.add_argument("--config", type=Path, default=HERE / "configs" / "smoke.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf8"))
    run_smoke(args.output, config)
    print(str((args.output / "report.json").resolve()))


if __name__ == "__main__":
    main()

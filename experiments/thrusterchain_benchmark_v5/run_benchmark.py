"""Reproducible integration profile; outputs are exclusive, never overwritten."""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
import scipy
import gymnasium
from thrusterchain.datasets import generate_dataset, load_dataset
from thrusterchain.regular import (train_behavior_cloning, freeze_policy, load_frozen_policy,
    calibrate_frozen_policy, evaluate_frozen_policy, FAMILIES)
from thrusterchain.planning import BoundedPlanner, identify_from_training, validation_residuals, array_hash
from thrusterchain.environment import ThrusterChainEnv, Physics, matrices
from thrusterchain.inference_guard import InferenceGuard
from thrusterchain.reporting import write_control_csv

ROOT = Path(__file__).resolve().parent

def write_json(path, value):
    with Path(path).open("x", encoding="utf8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")

def source_manifest():
    files = {}
    for parent in ("thrusterchain","tests","host_integration"):
        for p in sorted((ROOT/parent).rglob("*")):
            if not p.is_file() or p.suffix not in (".py",".json"): continue
            files[p.relative_to(ROOT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted((ROOT/"contracts").glob("*.json")):
        files[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(list(ROOT.glob("*.py"))+[ROOT/"pyproject.toml"]):
        if p.exists(): files[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    encoded=json.dumps(files,sort_keys=True,separators=(",",":")).encode()
    return dict(files=files,source_sha256=hashlib.sha256(encoded).hexdigest(),
                python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
                gymnasium=gymnasium.__version__,platform=platform.platform())

def export_transitions(dataset, path):
    """D4RL-style field names with explicit within-episode successor identity."""
    a=dataset.arrays; n,t=a["actions"].shape[:2]
    next_actions=np.zeros_like(a["actions"]); next_actions[:,:-1]=a["actions"][:,1:]
    valid=np.ones((n,t),bool); valid[:,-1]=False
    arrays=dict(observations=a["observations"][:,:-1].reshape(-1,14),
        next_observations=a["observations"][:,1:].reshape(-1,14),
        actions=a["actions"].reshape(-1,4),rewards=a["rewards"].reshape(-1),
        terminals=a["terminated"].reshape(-1),timeouts=a["truncated"].reshape(-1),
        next_actions=next_actions.reshape(-1,4),next_action_valid=valid.reshape(-1),
        episode_id=np.repeat(a["trajectory_ids"],t),step_id=np.tile(np.arange(t),n),
        split_code=np.repeat(a["split_codes"],t))
    with Path(path).open("xb") as f: np.savez_compressed(f,**arrays)
    return dict(dataset_sha256=dataset.digest,ordered_transitions=True,
        episode_boundaries_complete=True,
        observation_order=("physical13,remaining_fraction" if dataset.metadata["task_semantics"] == "finite_horizon"
                           else "physical13,constant_zero_padding"),
        next_observation_semantics="post_step_not_autoreset",
        next_action_semantics="same episode step+1; boundary masked invalid",
        terminal_semantics=dataset.metadata["task_semantics"],action_low=[-1]*4,action_high=[1]*4,
        normalization={"kind":"none"},source="run_benchmark.export_transitions",
        score_normalization="none; raw and gamma=.98 discounted returns",
        minari_package_compatibility_tested=False)

def evaluate_reference(controller, receipt, seeds, physics):
    if controller is not None and type(controller) is not BoundedPlanner:
        raise TypeError("reference evaluator accepts only frozen BoundedPlanner")
    if controller is not None and any(receipt.get(k)!=v for k,v in controller.receipt().items()):
        raise ValueError("live controller does not match frozen reference receipt")
    before=array_hash(controller.A,controller.B) if controller else None
    config_before=json.dumps(controller.receipt(),sort_keys=True) if controller else None
    planning_seconds=0.0
    families={}; certificates=[]; trace=None
    for family in FAMILIES:
        returns=[]; disc=[]; stalls=[]
        env=ThrusterChainEnv(**physics,start=family)
        for episode_seed in seeds:
            obs, info=env.reset(seed=episode_seed)
            rng=np.random.default_rng([episode_seed,812])
            total=discount=0.; count=0
            points=[{"t":0,"state":info["state"].tolist()}]
            for step in range(100):
                started=time.perf_counter()
                action=controller.act(obs) if controller else rng.uniform(-1,1,4).astype(np.float32)
                planning_seconds+=time.perf_counter()-started
                if controller: certificates.append(dict(controller.last_certificate))
                obs,reward,term,trunc,info=env.step(action)
                if (term,trunc)!=(step==99,False): raise RuntimeError("finite task boundary mismatch")
                total+=reward; discount+=.98**step*reward
                count+=int(np.count_nonzero(info["stalled"]))
                points.append({"t":step+1,"state":info["state"].tolist(),"action":action.tolist(),"reward":reward})
            returns.append(total);disc.append(discount);stalls.append(count/400)
            if trace is None: trace={"family":family,"seed":episode_seed,"points":points}
        env.close()
        families[family]=dict(episodes=len(seeds),episode_seeds=seeds,returns=returns,
            discounted_returns=disc,mean_return=float(np.mean(returns)),mean_discounted_return=float(np.mean(disc)),
            sample_sd_return=float(np.std(returns,ddof=1)) if len(returns)>1 else None,
            stall_fractions=stalls,held_out_from_dataset_generation=family!="recorded")
    after=array_hash(controller.A,controller.B) if controller else None
    if before!=after: raise RuntimeError("model parameters changed during evaluation")
    if controller and config_before!=json.dumps(controller.receipt(),sort_keys=True):
        raise RuntimeError("inference configuration changed during evaluation")
    return dict(receipt=receipt,model_hash_before=before,model_hash_after=after,
        families=families,parameter_updates=0,normalization_updates=0,
        inference_seconds=planning_seconds,solver_failures=0,fallback_actions=0,
        max_solver_kkt=max([x["projected_gradient_residual"] for x in certificates],default=None),
        every_solve_full_remaining_horizon=all(x["full_remaining_horizon"] for x in certificates) if controller else None,
        trace=trace)

def run(output, episodes=40, evaluation_episodes=2, planner_horizon=12, seed=20261022):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/"source_manifest_before.json",source_manifest())
    # The fixed offline data is fully materialized before fitting.
    generate_dataset(output/"offline_dataset.npz",episodes=episodes,seed=seed,provenance="smoke")
    dataset=load_dataset(output/"offline_dataset.npz")
    write_json(output/"dataset_manifest.json",dict(dataset_sha256=dataset.digest,metadata=dataset.metadata))
    write_json(output/"transition_export.json",export_transitions(dataset,output/"transitions.npz"))
    policy,selection=train_behavior_cloning(dataset)
    freeze_policy(policy,selection,output/"frozen_bc")
    policy,selection=load_frozen_policy(output/"frozen_bc")
    write_json(output/"selection_receipt.json",dict(sha256=selection.digest,payload=selection.payload))
    learned=identify_from_training(dataset)
    from thrusterchain.training_protocol import DEFAULT_OFFLINE_UPDATES, MAX_OFFLINE_UPDATES
    write_json(output/"training_budget_receipts.json",{
        "schema":"thrusterchain.closed_form_benchmark_budgets.v1",
        "mode":"strict_offline","frozen_policy":True,
        "default_iterative_primary_optimizer_budget":DEFAULT_OFFLINE_UPDATES,
        "maximum_iterative_primary_optimizer_budget":MAX_OFFLINE_UPDATES,
        "iterative_budget_applicability":"not_applicable_to_these_closed_form_and_reference_methods",
        "methods":{
            "behavior_cloning":{"fit_semantics":"closed_form_candidate_least_squares",
                "candidate_fits":selection.payload["budget"]["candidate_fits"],
                "primary_optimizer":None,"executed_primary_optimizer_updates":0,
                "executed_optimizer_steps_total":0,"online_adaptation_supported":False},
            "identified_model_mpc":{"fit_semantics":"train_only_closed_form_system_identification",
                "fits":1,"primary_optimizer":None,"executed_primary_optimizer_updates":0,
                "executed_optimizer_steps_total":0,"online_adaptation_supported":False},
            "true_model_mpc":{"fit_semantics":"evaluator_privileged_reference_no_fit",
                "executed_optimizer_steps_total":0},
            "random":{"fit_semantics":"no_fit","executed_optimizer_steps_total":0}},
        "online_training_environment_interactions":0,
        "frozen_evaluation_episodes_per_method_per_family":evaluation_episodes})
    write_json(output/"identification_validation.json",validation_residuals(dataset,learned))
    physics=Physics(**dataset.metadata["physics"])
    oracle=BoundedPlanner(physics=physics,planning_horizon=planner_horizon)
    identified=BoundedPlanner(learned.A,learned.B,physics,planner_horizon,
                             provenance="train_only_identified_model")
    np.savez_compressed(output/"identified_model.npz",A=learned.A,B=learned.B)
    reference_receipts={
        "random":{"kind":"uniform_random","seed_rule":"SeedSequence([episode_seed,812])","action_bounds":[-1,1],
                  "parameter_updates":0,"selection":"predeclared"},
        "true_model_mpc":oracle.receipt(),
        "identified_model_mpc":{**identified.receipt(),"identification":learned.receipt}}
    write_json(output/"reference_freeze_receipts.json",reference_receipts)
    calibration=calibrate_frozen_policy(dataset,policy,selection)
    write_json(output/"imitation_final_calibration.json",calibration)
    # All fit, selection, receipts and calibration above precede live evaluation.
    with InferenceGuard(activate_context=False) as guard:
        bc=evaluate_frozen_policy(policy,selection,episodes_per_family=evaluation_episodes,seed=seed+100000)
    bc["inference_guard"]=guard.receipt()
    seeds=bc["families"]["recorded"]["episode_seeds"]
    baselines={"behavior_cloning":bc}
    for name,controller in (("random",None),("true_model_mpc",oracle),("identified_model_mpc",identified)):
        with InferenceGuard() as guard:
            baselines[name]=evaluate_reference(controller,reference_receipts[name],seeds,dataset.metadata["physics"])
        baselines[name]["inference_guard"]=guard.receipt()
    if load_dataset(output/"offline_dataset.npz").digest!=dataset.digest:
        raise RuntimeError("evaluation mutated offline dataset")
    from thrusterchain.calibration import nine_cell_smoke
    diagnostic=nine_cell_smoke()
    write_json(output/"diagnostic_results.json",diagnostic)
    results=dict(schema="thrusterchain.benchmark_results.v1",profile="smoke",
        research_evidence=False,newly_trained_nine_cell_policy_grid_complete=False,
        training_episodes=episodes,evaluation_episodes_per_family=evaluation_episodes,
        planner_horizon=planner_horizon,seed=seed,dataset_sha256=dataset.digest,
        baselines=baselines,diagnostic_results="diagnostic_results.json",
        interpretation=["Integration demonstration only; one fitted training seed.",
            "MPC uses a shorter horizon unless explicitly set to 100; no full-episode optimum claim.",
            "Identified model shares the true-model planner and knows the public actuator/reward structure.",
            "Raw return higher (less negative) is better; no D4RL normalized-score claim.",
            "Historical 45-run BCA outcomes are independent development evidence; no BCA win is asserted."])
    write_control_csv(results, output/"control_episode_records.csv")
    after_manifest=source_manifest()
    write_json(output/"source_manifest_after.json",after_manifest)
    if json.loads((output/"source_manifest_before.json").read_text())["source_sha256"] != after_manifest["source_sha256"]:
        write_json(output/"INVALID_RUN.json",{"reason":"source changed during execution","research_evidence":False})
        raise RuntimeError("source changed during run; run is not final evidence")
    write_json(output/"benchmark_results.json",results)
    print(json.dumps({"output":str(output.resolve()),"dataset_sha256":dataset.digest,
                     "profiles":"Regular live evaluation + nine-cell diagnostic smoke",
                     "research_evidence":False}))
    return results

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--episodes",type=int,default=40)
    parser.add_argument("--evaluation-episodes",type=int,default=2)
    parser.add_argument("--planner-horizon",type=int,default=12)
    parser.add_argument("--seed",type=int,default=20261022)
    args=parser.parse_args()
    run(args.output,args.episodes,args.evaluation_episodes,args.planner_horizon,args.seed)


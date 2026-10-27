"""Stage-separated offline training and explicit offline-to-online experiments."""
import argparse
import importlib
import json
from pathlib import Path
import numpy as np
from .datasets import SCHEMA as ORIGINAL_SCHEMA, DatasetError, generate_dataset, load_dataset
from .quality_datasets import (BEHAVIORS, SCHEMA as BEHAVIOR_SCHEMA,
                               generate_behavior_dataset, load_behavior_dataset)
from .regular import (train_behavior_cloning,freeze_policy,load_frozen_policy,
    calibrate_frozen_policy,evaluate_frozen_policy)

HOST_ADAPTERS = {
    "td3bc": ("host_integration.td3bc_adapter", "TD3BCAdapter", "repository_td3bc_strict_v6_1"),
    "iql": ("host_integration.iql_adapter", "IQLAdapter", "repository_iql_strict_v6_1"),
    "cql": ("host_integration.cql_adapter", "CQLAdapter", "repository_cql_strict_v6_1"),
    "rebrac": ("host_integration.rebrac_adapter", "ReBRACAdapter", "repository_rebrac_strict_v6_1"),
    "unifloral-unified": ("host_integration.unifloral_adapter", "UnifiedTD3BCAdapter", "unifloral_unified_td3bc_outer_v6_1"),
}


def iterative_factory(name):
    """Load optional neural dependencies only for an explicitly selected host."""
    if name == "affine-sgd":
        from .sgd_adapter import AffineSGDBCAdapter
        return AffineSGDBCAdapter
    if name not in HOST_ADAPTERS:
        raise ValueError("unknown iterative adapter")
    module, cls, _ = HOST_ADAPTERS[name]
    return getattr(importlib.import_module(module), cls)


def checkpoint_factory(adapter_id):
    if adapter_id == "affine_sgd_bc_protocol_v1":
        return iterative_factory("affine-sgd")
    for name, (_, _, identity) in HOST_ADAPTERS.items():
        if adapter_id == identity:
            return iterative_factory(name)
    raise ValueError("unsupported checkpoint adapter identity")

def save(path,record):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf8") as f: json.dump(record,f,indent=2,allow_nan=False)

def settings(text):
    value=json.loads(text)
    if type(value) is not dict:
        raise ValueError("learning settings must be a JSON object")
    return value

def check_new_output(path):
    """Reject existing targets and file ancestors before any expensive work."""
    path=Path(path)
    if path.exists():
        raise FileExistsError(path)
    ancestor=path.parent
    while not ancestor.exists():
        ancestor=ancestor.parent
    if not ancestor.is_dir():
        raise NotADirectoryError(ancestor)


def load_benchmark_dataset(path):
    """Dispatch only on an explicit schema; each loader validates every field."""
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            manifest=archive["manifest"]
            if manifest.shape != () or manifest.dtype.kind != "U":
                raise DatasetError("dataset manifest must be a scalar Unicode JSON string")
            envelope=json.loads(str(manifest))
        if not isinstance(envelope, dict) or not isinstance(envelope.get("metadata"), dict):
            raise DatasetError("invalid dataset manifest envelope")
        schema=envelope["metadata"].get("schema")
        if schema == ORIGINAL_SCHEMA:
            return load_dataset(path)
        if schema == BEHAVIOR_SCHEMA:
            return load_behavior_dataset(path)
        raise DatasetError("unsupported dataset schema")
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise DatasetError(f"cannot inspect dataset manifest: {exc}") from exc

def train_stage(args):
    from .training_protocol import OfflineBudget, DEFAULT_OFFLINE_UPDATES
    budget=OfflineBudget(updates=DEFAULT_OFFLINE_UPDATES if args.offline_updates is None else args.offline_updates,
                         batch_size=args.batch_size)
    check_new_output(args.frozen_directory)
    if args.adapter=="closed-form-bc":
        if args.offline_updates is not None or args.resume_checkpoint is not None or args.learning_settings is not None:
            raise ValueError("closed-form BC has no optimizer budget, online adaptation or optimizer resume; use --adapter affine-sgd for the iterative protocol example")
        d=load_benchmark_dataset(args.dataset);policy,receipt=train_behavior_cloning(d)
        frozen=freeze_policy(policy,receipt,args.frozen_directory)
        result={"schema":"thrusterchain.closed_form_budget.v1", "adapter":"closed-form-bc",
                "fit_semantics":"closed_form_candidate_least_squares", "frozen":str(frozen),
                "dataset_sha256":d.digest,"policy_sha256":policy.digest,"receipt_sha256":receipt.digest,
                "mode":"strict_offline", "default_offline_optimizer_budget":budget.updates,
                "budget_applicability":"not_applicable", "executed_optimizer_updates":0,
                "candidate_fits":receipt.payload["budget"]["candidate_fits"],
                "online_environment_interactions":0,"evaluation_episodes":0,"simulator_selection_steps":0}
        save(args.frozen_directory/"offline_budget_receipt.json",result)
        return result
    from .training_protocol import LearnerCheckpoint
    from .training_runner import train_offline
    if args.adapter in HOST_ADAPTERS and args.offline_updates is None:
        raise ValueError("neural adapters require an explicit --offline-updates budget")
    d=load_benchmark_dataset(args.dataset)
    resume=LearnerCheckpoint.load(args.resume_checkpoint) if args.resume_checkpoint else None
    checkpoint,result=train_offline(d,iterative_factory(args.adapter),budget=budget,seed=args.seed,
        learning_settings=settings(args.learning_settings) if args.learning_settings is not None else None,
        resume_from=resume)
    args.frozen_directory.mkdir(parents=True,exist_ok=False)
    checkpoint.save(args.frozen_directory/"checkpoint.json")
    save(args.frozen_directory/"offline_training_receipt.json",result)
    return result

def evaluation_stage(args):
    from .training_protocol import EvaluationConfig, OnlineBudget, LearnerCheckpoint
    supplied=(args.online_interactions,args.online_updates,args.offline_fraction,args.learning_settings)
    if args.frozen_policy:
        if any(value is not None for value in supplied):
            raise ValueError("strict frozen evaluation rejects online learning settings and budgets")
        online=None
    else:
        if args.learning is not None:
            raise ValueError("--learning disabled is incompatible with --no-frozen-policy; omit the legacy flag for adaptation")
        if any(value is None for value in supplied):
            raise ValueError("--no-frozen-policy requires --online-interactions, --online-updates, --offline-fraction and --learning-settings")
        online=OnlineBudget(interactions=args.online_interactions,updates=args.online_updates,
            offline_fraction=args.offline_fraction,batch_size=args.batch_size,
            learning_settings=settings(args.learning_settings))
    config=EvaluationConfig(frozen_policy=args.frozen_policy,episodes=args.episodes,seed=args.seed,online=online)
    check_new_output(args.output)
    if (args.frozen_directory/"checkpoint.json").is_file():
        from .training_runner import run_mode
        if args.dataset is None:
            raise ValueError("the iterative runner requires --dataset for the fixed-data identity and Train replay view")
        checkpoint_dir=args.output.with_name(args.output.stem+"_checkpoints")
        check_new_output(checkpoint_dir)
        checkpoint=LearnerCheckpoint.load(args.frozen_directory/"checkpoint.json")
        result=run_mode(load_benchmark_dataset(args.dataset),checkpoint,checkpoint_factory(checkpoint.adapter_id),config,
                        checkpoint_directory=checkpoint_dir)
        # The runner saved and reloaded endpoints before any frozen scoring episodes.
        artifacts={}
        for family,cp in result["checkpoints"].items():
            path=checkpoint_dir/(family+".json")
            if LearnerCheckpoint.load(path).digest!=cp.digest:
                raise ValueError("persisted checkpoint changed after frozen evaluation")
            artifacts[family]={"path":str(path),"sha256":cp.digest}
        result={**result,"checkpoints":artifacts}
    else:
        if not args.frozen_policy:
            raise ValueError("closed-form BC is inference-only and does not support offline-to-online adaptation")
        policy,receipt=load_frozen_policy(args.frozen_directory)
        from .inference_guard import InferenceGuard
        with InferenceGuard(activate_context=False) as guard:
            result=evaluate_frozen_policy(policy,receipt,episodes_per_family=args.episodes,seed=args.seed)
        result["inference_guard"]=guard.receipt()
        result["mode"]="strict_offline"
        result["frozen_policy"]=True
        result["online_optimizer_updates"]=0
    save(args.output,result)
    return result


def shift_evaluation_stage(args):
    """Evaluate a dataset-bound frozen iterative checkpoint on v6 shifts."""
    from .training_protocol import LearnerCheckpoint
    from .shift_scenarios_v1 import evaluate_frozen_shifts
    check_new_output(args.output)
    dataset = load_benchmark_dataset(args.dataset)
    if dataset.metadata["task_semantics"] != "finite_horizon":
        raise ValueError("v6 shift evaluation requires a finite-horizon dataset and policy")
    checkpoint = LearnerCheckpoint.load(args.frozen_directory / "checkpoint.json")
    runner_state = checkpoint.metadata.get("runner_state", {})
    if (runner_state.get("dataset_sha256") != dataset.digest
            or runner_state.get("train_split_sha256") != dataset.split_digest("train")):
        raise ValueError("checkpoint must be bound to the specified dataset and training split")
    learner = checkpoint_factory(checkpoint.adapter_id)()
    learner.restore(checkpoint)
    before = learner.checkpoint().digest
    policy = learner.freeze()
    result = evaluate_frozen_shifts(policy, scenarios=args.scenarios,
        partitions={"dataset": dataset.arrays["episode_seeds"].tolist(),
                    "selection": args.selection_seeds, "final": args.final_seeds},
        purpose=args.purpose, policy_task_semantics="finite_horizon",
        stall=dataset.metadata["physics"]["stall"],
        stall_slope=dataset.metadata["physics"]["stall_slope"])
    if learner.checkpoint().digest != before:
        raise ValueError("frozen shift evaluation mutated the learner")
    result.update(dataset_sha256=dataset.digest, checkpoint_sha256=checkpoint.digest,
                  training_split_sha256=dataset.split_digest("train"))
    save(args.output, result)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    g=sub.add_parser("generate",help="collect a new fixed continuous offline episode dataset")
    g.add_argument("dataset",type=Path);g.add_argument("--episodes",type=int,default=40)
    g.add_argument("--seed",type=int,default=20261022);g.add_argument("--epsilon",type=float,default=.25)
    b=sub.add_parser("generate-behavior",help="collect a new versioned random/feedback trajectory dataset")
    b.add_argument("dataset",type=Path);b.add_argument("--behavior",choices=BEHAVIORS,required=True)
    b.add_argument("--episodes",type=int,default=40);b.add_argument("--seed",type=int,default=20261027)
    b.add_argument("--stall",action=argparse.BooleanOptionalAction,default=True)
    b.add_argument("--stall-slope",type=float,choices=(1.5,3.0),default=1.5)
    b.add_argument("--task-semantics",choices=("finite_horizon","continuing_time_limit"),default="finite_horizon")
    v=sub.add_parser("validate",help="validate dataset schema, dynamics and digest")
    v.add_argument("dataset",type=Path)
    t=sub.add_parser("train",help="train from fixed data; default closed-form BC or iterative protocol example")
    t.add_argument("dataset",type=Path);t.add_argument("frozen_directory",type=Path)
    t.add_argument("--adapter",choices=["closed-form-bc","affine-sgd",*HOST_ADAPTERS],default="closed-form-bc")
    t.add_argument("--offline-updates",type=int,default=None,help="primary optimizer steps: iterative default 1000000, cumulative maximum 3000000; inapplicable to closed-form BC")
    t.add_argument("--batch-size",type=int,default=256)
    t.add_argument("--seed",type=int,default=20261022)
    t.add_argument("--learning-settings",help="explicit JSON object of adapter settings")
    t.add_argument("--resume-checkpoint",type=Path)
    c=sub.add_parser("calibrate",help="fresh final imitation calibration after freezing")
    c.add_argument("dataset",type=Path);c.add_argument("frozen_directory",type=Path)
    c.add_argument("output",type=Path)
    e=sub.add_parser("evaluate",help="frozen evaluation by default; explicit toggle for a separate adaptation experiment")
    e.add_argument("frozen_directory",type=Path);e.add_argument("output",type=Path)
    e.add_argument("--episodes",type=int,default=2);e.add_argument("--seed",type=int,default=20361022)
    e.add_argument("--learning",choices=["disabled"],default=None,help="legacy strict-evaluation flag; incompatible with adaptation")
    e.add_argument("--frozen-policy",action=argparse.BooleanOptionalAction,default=True)
    e.add_argument("--dataset",type=Path)
    e.add_argument("--online-interactions",type=int)
    e.add_argument("--online-updates",type=int)
    e.add_argument("--offline-fraction",type=float)
    e.add_argument("--batch-size",type=int,default=256)
    e.add_argument("--learning-settings")
    s=sub.add_parser("evaluate-shifts", help="evaluate a dataset-bound frozen finite-task checkpoint on v6 shifts")
    s.add_argument("frozen_directory",type=Path);s.add_argument("output",type=Path)
    s.add_argument("--dataset",type=Path,required=True)
    s.add_argument("--scenarios",nargs="+",required=True,
                   choices=("baseline","attenuated_thruster","bounded_disturbance","combined"))
    s.add_argument("--final-seeds",nargs="+",type=int,required=True)
    s.add_argument("--selection-seeds",nargs="*",type=int,default=[])
    s.add_argument("--purpose",choices=("selection","final"),default="final")
    args=p.parse_args()
    if args.command=="generate":
        d=generate_dataset(args.dataset,episodes=args.episodes,seed=args.seed,epsilon=args.epsilon,provenance="smoke")
        result={"dataset":str(args.dataset),"sha256":d.digest,"profile":"smoke"}
    elif args.command=="generate-behavior":
        check_new_output(args.dataset)
        d=generate_behavior_dataset(args.dataset,behavior=args.behavior,episodes=args.episodes,seed=args.seed,
                                    stall=args.stall,stall_slope=args.stall_slope,
                                    task_semantics=args.task_semantics,provenance="smoke")
        result={"dataset":str(args.dataset),"sha256":d.digest,"schema":BEHAVIOR_SCHEMA,"profile":"smoke"}
    elif args.command=="validate":
        d=load_benchmark_dataset(args.dataset);result={"valid":True,"sha256":d.digest,"metadata":d.metadata}
    elif args.command=="train":
        result=train_stage(args)
    elif args.command=="calibrate":
        d=load_benchmark_dataset(args.dataset);policy,receipt=load_frozen_policy(args.frozen_directory)
        result=calibrate_frozen_policy(d,policy,receipt);save(args.output,result)
    elif args.command=="evaluate-shifts":
        result=shift_evaluation_stage(args)
    else:
        result=evaluation_stage(args)
    print(json.dumps(result,allow_nan=False))
if __name__=="__main__":main()


"""Stage-separated offline training and explicit offline-to-online experiments."""
import argparse
import json
from pathlib import Path
from .datasets import generate_dataset, load_dataset
from .regular import (train_behavior_cloning,freeze_policy,load_frozen_policy,
    calibrate_frozen_policy,evaluate_frozen_policy)

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

def train_stage(args):
    from .training_protocol import OfflineBudget, DEFAULT_OFFLINE_UPDATES
    budget=OfflineBudget(updates=DEFAULT_OFFLINE_UPDATES if args.offline_updates is None else args.offline_updates,
                         batch_size=args.batch_size)
    check_new_output(args.frozen_directory)
    if args.adapter=="closed-form-bc":
        if args.offline_updates is not None or args.resume_checkpoint is not None or args.learning_settings is not None:
            raise ValueError("closed-form BC has no optimizer budget, online adaptation or optimizer resume; use --adapter affine-sgd for the iterative protocol example")
        d=load_dataset(args.dataset);policy,receipt=train_behavior_cloning(d)
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
    from .sgd_adapter import AffineSGDBCAdapter
    d=load_dataset(args.dataset)
    resume=LearnerCheckpoint.load(args.resume_checkpoint) if args.resume_checkpoint else None
    checkpoint,result=train_offline(d,AffineSGDBCAdapter,budget=budget,seed=args.seed,
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
        from .sgd_adapter import AffineSGDBCAdapter
        if args.dataset is None:
            raise ValueError("the iterative runner requires --dataset for the fixed-data identity and Train replay view")
        checkpoint_dir=args.output.with_name(args.output.stem+"_checkpoints")
        check_new_output(checkpoint_dir)
        checkpoint=LearnerCheckpoint.load(args.frozen_directory/"checkpoint.json")
        if checkpoint.adapter_id!="affine_sgd_bc_protocol_v1":
            raise ValueError("CLI supports only affine_sgd_bc_protocol_v1; external hosts use the Python factory interface")
        result=run_mode(load_dataset(args.dataset),checkpoint,AffineSGDBCAdapter,config,
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

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    g=sub.add_parser("generate",help="collect a new fixed continuous offline episode dataset")
    g.add_argument("dataset",type=Path);g.add_argument("--episodes",type=int,default=40)
    g.add_argument("--seed",type=int,default=20261022);g.add_argument("--epsilon",type=float,default=.25)
    v=sub.add_parser("validate",help="validate dataset schema, dynamics and digest")
    v.add_argument("dataset",type=Path)
    t=sub.add_parser("train",help="train from fixed data; default closed-form BC or iterative protocol example")
    t.add_argument("dataset",type=Path);t.add_argument("frozen_directory",type=Path)
    t.add_argument("--adapter",choices=["closed-form-bc","affine-sgd"],default="closed-form-bc")
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
    args=p.parse_args()
    if args.command=="generate":
        d=generate_dataset(args.dataset,episodes=args.episodes,seed=args.seed,epsilon=args.epsilon,provenance="smoke")
        result={"dataset":str(args.dataset),"sha256":d.digest,"profile":"smoke"}
    elif args.command=="validate":
        d=load_dataset(args.dataset);result={"valid":True,"sha256":d.digest,"metadata":d.metadata}
    elif args.command=="train":
        result=train_stage(args)
    elif args.command=="calibrate":
        d=load_dataset(args.dataset);policy,receipt=load_frozen_policy(args.frozen_directory)
        result=calibrate_frozen_policy(d,policy,receipt);save(args.output,result)
    else:
        result=evaluation_stage(args)
    print(json.dumps(result,allow_nan=False))
if __name__=="__main__":main()


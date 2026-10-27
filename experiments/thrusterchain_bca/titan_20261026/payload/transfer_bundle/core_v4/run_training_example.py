"""Small real-update protocol smoke; no full-budget research sweep."""
import argparse
import hashlib
from pathlib import Path

from run_benchmark import source_manifest, write_json
from thrusterchain.datasets import load_dataset
from thrusterchain.training_protocol import OfflineBudget, OnlineBudget, EvaluationConfig, LearnerCheckpoint
from thrusterchain.training_runner import train_offline, run_mode
from thrusterchain.sgd_adapter import AffineSGDBCAdapter


def save_mode(directory, result):
    directory.mkdir(parents=True,exist_ok=True)
    checkpoints={}
    for family,checkpoint in result["checkpoints"].items():
        path=directory/"checkpoints"/(family+".json")
        assert LearnerCheckpoint.load(path).digest==checkpoint.digest
        checkpoints[family]={"path":path.relative_to(directory).as_posix(),"sha256":checkpoint.digest}
    write_json(directory/"results.json",{**result,"checkpoints":checkpoints})


def run(output, dataset_path):
    output=Path(output)
    dataset=load_dataset(dataset_path)
    data_file_hash=hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()
    output.mkdir(parents=True,exist_ok=False)
    snapshot=source_manifest()
    write_json(output/"source_manifest_before.json",snapshot)
    config={"schema":"thrusterchain.training_example.v1","profile":"protocol_smoke",
        "offline_primary_updates":8,"resume_additional_primary_updates":3,"batch_size":16,
        "online_interactions_per_scenario":12,"online_updates_per_scenario":6,
        "online_offline_fraction":0.5,"evaluation_episodes_per_scenario":2,
        "default_offline_primary_budget":1000000,"maximum_offline_primary_budget":3000000,
        "full_default_or_maximum_budget_executed":False,
        "adapter":"affine_sgd_bc_protocol_v1","learning_settings":{},
        "dataset":str(Path(dataset_path).resolve()),"dataset_sha256":dataset.digest,
        "dataset_file_sha256":data_file_hash,"seed":20261022,
        "scientific_claim":"protocol integration; no BCA, neural RL or adaptation-benefit claim"}
    write_json(output/"registered_config.json",config)
    checkpoint,offline=train_offline(dataset,AffineSGDBCAdapter,budget=OfflineBudget(8,16),seed=20261022)
    checkpoint.save(output/"pretrained_checkpoint.json")
    write_json(output/"offline_training_receipt.json",offline)
    resumed,resume_receipt=train_offline(dataset,AffineSGDBCAdapter,budget=OfflineBudget(3,16),
        seed=20261022,resume_from=checkpoint)
    uninterrupted,whole_receipt=train_offline(dataset,AffineSGDBCAdapter,budget=OfflineBudget(11,16),seed=20261022)
    # Resume restores the sampler cursor and every learner component, not only weights.
    component_agreement=resumed.component_hashes==uninterrupted.component_hashes
    byte_agreement=resumed.to_bytes()==uninterrupted.to_bytes()
    assert component_agreement and byte_agreement and resumed.optimizer_steps==uninterrupted.optimizer_steps==11
    resumed.save(output/"resumed_checkpoint.json")
    write_json(output/"resume_receipt.json",resume_receipt)
    write_json(output/"resume_comparison.json",{
        "all_adapter_component_hashes_equal":component_agreement,
        "full_checkpoint_bytes_equal":byte_agreement,
        "resumed_component_hashes":resumed.component_hashes,
        "uninterrupted_component_hashes":uninterrupted.component_hashes,
        "executed_optimizer_steps":11,"uninterrupted_receipt":whole_receipt,
        "scope":"offline learner and minibatch-stream continuation; not mid-episode online resume"})
    strict=run_mode(dataset,checkpoint,AffineSGDBCAdapter,EvaluationConfig(episodes=2,seed=20361022),
                    checkpoint_directory=output/"strict"/"checkpoints")
    save_mode(output/"strict",strict)
    adaptive=run_mode(dataset,checkpoint,AffineSGDBCAdapter,EvaluationConfig(frozen_policy=False,
        episodes=2,seed=20361022,online=OnlineBudget(interactions=12,updates=6,offline_fraction=.5,
            batch_size=16,learning_settings={})),checkpoint_directory=output/"adaptation"/"checkpoints")
    save_mode(output/"adaptation",adaptive)
    zero=run_mode(dataset,checkpoint,AffineSGDBCAdapter,EvaluationConfig(frozen_policy=False,
        episodes=2,seed=20361022,online=OnlineBudget(interactions=0,updates=0,offline_fraction=.5,
            batch_size=16,learning_settings={})),checkpoint_directory=output/"zero_online_budget"/"checkpoints")
    save_mode(output/"zero_online_budget",zero)
    assert hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()==data_file_hash
    assert source_manifest()["source_sha256"]==snapshot["source_sha256"]
    write_json(output/"completion.json",{
        "source_sha256":snapshot["source_sha256"],"status":"pass","profile":"protocol_smoke",
        "offline_dataset_unchanged":True,"offline_resume_components_equal":component_agreement,
        "offline_resume_checkpoint_bytes_equal":byte_agreement,
        "pretrained_optimizer_steps":checkpoint.optimizer_steps,
        "adapted_optimizer_steps":{k:v.optimizer_steps for k,v in adaptive["checkpoints"].items()},
        "zero_budget_optimizer_steps":{k:v.optimizer_steps for k,v in zero["checkpoints"].items()},
        "modes":["strict_offline","offline_to_online","zero_online_budget"],
        "learned_neural_methods_evaluated":0})
    return output


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--dataset",required=True,type=Path)
    args=parser.parse_args()
    print(run(args.output,args.dataset))

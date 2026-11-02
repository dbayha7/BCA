"""Generate the predeclared isolated full-size *smoke* dataset on CPU."""
import hashlib
import json
from pathlib import Path
import time

from thrusterchain.quality_datasets import generate_behavior_dataset
from thrusterchain.training_runner import train_transitions

root = Path(__file__).parent / "resource_20261027_v1"
root.mkdir(exist_ok=False)
started = time.perf_counter()
dataset = generate_behavior_dataset(root / "data.npz", behavior="mixed_random_strong",
                                    episodes=1000, seed=62077, provenance="smoke")
train = train_transitions(dataset)
assert len(train) == 60000
assert dataset.metadata["horizon"] == 100
plan = Path(__file__).with_name("RESOURCE_RESUME_PLAN_20261027.md")
info = {"status": "complete", "dataset_sha256": dataset.digest,
        "dataset_file_sha256": hashlib.sha256((root / "data.npz").read_bytes()).hexdigest(),
        "train_split_sha256": dataset.split_digest("train"), "seed": 62077,
        "episodes": 1000, "horizon": 100, "train_transitions": len(train),
        "behavior": "mixed_random_strong", "provenance": "smoke",
        "collection_wall_seconds": time.perf_counter() - started,
        "plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
with (root / "data_receipt.json").open("x", encoding="utf-8") as stream:
    json.dump(info, stream, indent=2, allow_nan=False)
print(json.dumps(info, indent=2))

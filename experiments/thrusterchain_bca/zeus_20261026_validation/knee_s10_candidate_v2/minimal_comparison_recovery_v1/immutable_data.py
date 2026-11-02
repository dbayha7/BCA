"""Consume one authenticated, already validated dataset without regenerating it.

This is an explicit loader amendment, not a change to dataset_knee.validate.
Producer-side exact lineage/dynamics validation is reused for exactly the
recorded canonical content. WSL reset replay failed by one ULP in episode 16;
no tolerance is introduced and target-runtime regeneration is not certified.
"""
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
PRODUCER = HERE.parent / "formal_g2"
RUN = PRODUCER / "formal_g2_run/attempt_1"
DATASET = RUN / "knee_s10_v2_stall3.npz"
DATASET_MODULE = HERE.parents[1] / "knee_edge_ood_v3/dataset_knee.py"
ARCHIVE_SHA = "125131d77f54600497ce4a5d95abff885c1cefbf8ccbafd2d74fd4d52da4bdb2"
CONTENT_SHA = "c2a6d61665a539ad8740d191424f90aeea9973566f3c6325226d928161fee533"
MANIFEST_SHA = "e7be6e218b77623dd5c4c67791dbb1f000f3c4269f7bf30911d8ab34806de2be"
REPORT_SHA = "a9c99a2067ed2aacfe3848100791e0cadee859d86c7d07e1e901244ddab83efb"
COLLECTION_SHA = "bb256064091864270e14fb623b710704b0ba14735ad5660cc2a296db7dd7780f"
DATASET_MODULE_SHA = "e895a098f7ef4d09be68821e132b21da487d6fed82d6610e9ed5ebe8b2a0b1c7"
RECORDER_SHA = "6f75001cf667e4b2b2559636abf74dcb081a071ed6a289b28a79b7ac13f193de"
ENTROPY = 6846509946489100059
SCHEMA = "thrusterchain.knee_edge_trajectories.v1"
DTYPES = {
    "observations": "<f4", "states": "<f8", "actions": "<f4", "requested_actions": "<f8", "rewards": "<f8",
    "terminated": "|b1", "truncated": "|b1", "trajectory_ids": "<i8", "group_ids": "<i8", "episode_seeds": "<i8",
    "split_codes": "|u1", "reset_component": "|u1", "behavior_component": "|u1", "load_band": "|u1",
    "behavior_log_density": "<f8",
}
SHAPES = {
    "observations": (1280, 101, 14), "states": (1280, 101, 13), "actions": (1280, 100, 4),
    "requested_actions": (1280, 100, 4), "rewards": (1280, 100), "terminated": (1280, 100),
    "truncated": (1280, 100), "trajectory_ids": (1280,), "group_ids": (1280,), "episode_seeds": (1280,),
    "split_codes": (1280,), "reset_component": (1280,), "behavior_component": (1280, 100),
    "load_band": (1280, 100), "behavior_log_density": (1280, 100),
}
LIMITATIONS = [
    "Producer full validation is established by its sealed successful collection call chain, not a standalone validation receipt.",
    "The producer's 44-file manifest does not fully attest its historical runtime/dependency closure.",
    "Target-runtime exact reset regeneration failed by one ULP in episode 16; it is not reclassified as passing.",
    "No target-runtime reset, recorder or dynamics replay is performed by this loader; unchanged authenticated content is reused.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read_bound(path, expected):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, "Producer evidence changed: " + str(path))
    return json.loads(raw)


def authenticate_producer():
    manifest = read_bound(PRODUCER / "G2_FORMAL_MANIFEST.json", MANIFEST_SHA)
    report = read_bound(RUN / "report.json", REPORT_SHA)
    collection = read_bound(RUN / "collection.json", COLLECTION_SHA)
    pins = {name: digest for section in ("files", "upstream") for name, digest in manifest[section].items()}
    require(len(pins) == 44, "Unexpected producer source closure")
    for name, expected in pins.items():
        path = (PRODUCER / name).resolve()
        require(path.is_relative_to(REPO), "Producer path escapes repository")
        require(sha(path) == expected, "Producer source changed: " + name)
    expected_collection = {"path": DATASET.name, "sha256": CONTENT_SHA, "episodes": 1280,
                           "entropy": ENTROPY, "recorder_sha256": RECORDER_SHA,
                           "behavior": "knee_s10_wide_window_v1"}
    require(collection.get("stall3") == expected_collection, "Producer collection identity differs")
    require(report.get("collection") == collection and report.get("error") is None,
            "Producer collection did not finish successfully")
    require(report.get("manifest_sha256") == MANIFEST_SHA and
            report.get("integrity_after") == {"manifest_sha256": MANIFEST_SHA, "ok": True},
            "Producer integrity evidence differs")
    return {"manifest_sha256": MANIFEST_SHA, "report_sha256": REPORT_SHA,
            "collection_sha256": COLLECTION_SHA, "verified_source_files": len(pins),
            "validation_reused": "generate -> KneeDataset.__init__ -> validate(check_dynamics=True), before digest and save",
            "producer_validated_episodes": 1280, "producer_validated_transition_rows": 128000}


def content_digest(arrays, metadata):
    """The exact pinned canonical digest algorithm; no simulator import."""
    digest = hashlib.sha256(canonical_json(metadata).encode("utf8"))
    for name in sorted(arrays):
        value = np.asarray(arrays[name])
        require(not value.dtype.hasobject, "Object arrays are forbidden")
        header = canonical_json([name, value.dtype.str, list(value.shape)]).encode("utf8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def check_content(arrays, envelope):
    require(set(envelope) == {"metadata", "sha256"}, "Invalid manifest envelope")
    require(set(arrays) == set(DTYPES), "Array schema differs")
    metadata = envelope["metadata"]
    require(metadata.get("schema") == SCHEMA and metadata.get("seed") == ENTROPY and
            metadata.get("world") == "stall3" and metadata.get("provenance") == "research" and
            metadata.get("episodes") == 1280 and metadata.get("horizon") == 100 and
            metadata.get("gamma") == .98 and metadata.get("observation_mode") == "time_aware" and
            metadata.get("task_semantics") == "finite_horizon" and metadata.get("recorder_sha256") == RECORDER_SHA,
            "Dataset metadata identity differs")
    require(metadata.get("split_counts") == {"train": 768, "tuning_validation": 256,
                                             "selection": 128, "final_calibration": 128}, "Split counts differ")
    for name, value in arrays.items():
        require(value.shape == SHAPES[name] and value.dtype.str == DTYPES[name], name + ": shape/dtype differs")
        require(value.dtype.kind != "f" or bool(np.all(np.isfinite(value))), name + ": nonfinite value")
    digest = content_digest(arrays, metadata)
    require(envelope["sha256"] == CONTENT_SHA and digest == CONTENT_SHA, "Canonical content hash differs")
    return metadata, digest


def load_verified(path, dataset_module):
    """Return the exact expected dataset class with immutable, authenticated bytes.

    Its ordinary constructor replays the simulator and is intentionally not
    called. This exception applies only to the fixed previously validated
    archive; this function cannot accept another dataset or a relaxed digest.
    """
    path = Path(path).resolve()
    require(path == DATASET.resolve(), "Only the authorized dataset path is accepted")
    require(Path(dataset_module.__file__).resolve() == DATASET_MODULE.resolve() and
            sha(DATASET_MODULE) == DATASET_MODULE_SHA and dataset_module.SCHEMA == SCHEMA and
            dataset_module.DTYPES == DTYPES, "Dataset class/source/schema differs")
    provenance = authenticate_producer()
    require(sha(path) == ARCHIVE_SHA, "Dataset archive hash differs")
    with np.load(path, allow_pickle=False) as archive:
        require(len(archive.files) == len(DTYPES) + 1 and set(archive.files) == set(DTYPES) | {"manifest"},
                "Unexpected or duplicate NPZ members")
        envelope = json.loads(str(archive["manifest"]))
        arrays = {name: archive[name] for name in DTYPES}
    metadata, digest = check_content(arrays, envelope)
    # bytes-backed arrays cannot be made writable again; retain every stored bit.
    immutable = {name: np.frombuffer(np.ascontiguousarray(value).tobytes(), dtype=value.dtype).reshape(value.shape)
                 for name, value in arrays.items()}
    result = object.__new__(dataset_module.KneeDataset)
    object.__setattr__(result, "_arrays", MappingProxyType(immutable))
    object.__setattr__(result, "_metadata_json", canonical_json(metadata))
    object.__setattr__(result, "_digest", digest)
    require(sha(path) == ARCHIVE_SHA, "Dataset changed while loading")
    authenticate_producer()
    return result, {"loader": "exact_immutable_artifact_reuse.v1", "archive_path": str(path),
                    "archive_sha256": ARCHIVE_SHA, "canonical_content_sha256": digest,
                    "dataset_module_sha256": DATASET_MODULE_SHA, "producer": provenance,
                    "array_count": len(immutable), "arrays_immutable": True, "metadata_unchanged": True,
                    "simulator_steps": 0, "resets": 0, "training_updates": 0, "calibrations": 0,
                    "historical_validation_work_reused_not_reexecuted": True, "limitations": LIMITATIONS}

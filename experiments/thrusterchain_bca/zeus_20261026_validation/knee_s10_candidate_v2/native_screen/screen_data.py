"""Change 7 (SCREEN_RULE.md, amendment A1): the two formal G2 v2 datasets are read without regenerating them.

`dataset_knee.load` validates an archive by resetting every episode again and replaying the recorder and the
physics, and demands bit-for-bit equality with the recorded arrays. The formal G2 collected and validated both
datasets in the Windows venv; the recorded Linux environment's reset of episode 16 reproduces `wave_1` one ULP off
(`../reset_lineage_diagnostic_v1/receipt_20261101T092712_649674_EST_424.json`), so that check cannot pass there.
No tolerance is introduced. Instead this loader authenticates, by sha256, the archive bytes, the canonical content
digest (the pinned algorithm of the core datasets module, recomputed here), and the producer's sealed evidence that
the full validation ran once and passed (`G2_FORMAL_MANIFEST.json`, the attempt-1 `report.json` with headline PASS
and `collection.json`, every one of the formal package's pinned source files), and returns the same immutable
`KneeDataset` class the pipeline expects. It accepts exactly the two recorded archives and nothing else.
Linux regeneration is not certified by this loader; it is recorded as not bit-portable.
"""
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
CAND = HERE.parent
F2 = CAND / "formal_g2"
RUN = F2 / "formal_g2_run" / "attempt_1"
for _p in (str(CAND.parent / "knee_edge_ood_v3"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import dataset_knee as D                                    # noqa: E402
from core import canonical_json, load_datasets, require, sha_file   # noqa: E402

# The producer's sealed evidence, by sha256 (each a constant of this package; the screen's manifest hashes the files
# too, so a change is refused twice).
MANIFEST_SHA = "e7be6e218b77623dd5c4c67791dbb1f000f3c4269f7bf30911d8ab34806de2be"
REPORT_SHA = "a9c99a2067ed2aacfe3848100791e0cadee859d86c7d07e1e901244ddab83efb"
COLLECTION_SHA = "bb256064091864270e14fb623b710704b0ba14735ad5660cc2a296db7dd7780f"
RECORDER_SHA = "6f75001cf667e4b2b2559636abf74dcb081a071ed6a289b28a79b7ac13f193de"
ENTROPY = 6846509946489100059
ARCHIVES = {                                                # archive bytes and canonical content, per world
    "stall3": {"file": "knee_s10_v2_stall3.npz",
               "archive_sha256": "125131d77f54600497ce4a5d95abff885c1cefbf8ccbafd2d74fd4d52da4bdb2",
               "content_sha256": "c2a6d61665a539ad8740d191424f90aeea9973566f3c6325226d928161fee533"},
    "nostall": {"file": "knee_s10_v2_nostall.npz",
                "archive_sha256": "5f4933fc5d687fc0b6f60e85c96809e7f0ace7023b91209ab89a9e373111b71e",
                "content_sha256": "e3549869af13b367cd726d008935b3eb466e38c005d1d5b7856faae3397e572a"},
}
SHAPES = {
    "observations": (1280, 101, 14), "states": (1280, 101, 13), "actions": (1280, 100, 4),
    "requested_actions": (1280, 100, 4), "rewards": (1280, 100), "terminated": (1280, 100),
    "truncated": (1280, 100), "trajectory_ids": (1280,), "group_ids": (1280,), "episode_seeds": (1280,),
    "split_codes": (1280,), "reset_component": (1280,), "behavior_component": (1280, 100),
    "load_band": (1280, 100), "behavior_log_density": (1280, 100),
}
LIMITATIONS = (
    "The producer's validation is established by its sealed, successful run record, not by a standalone receipt.",
    "The recorded Linux environment's exact reset regeneration failed by one ULP in episode 16; it is not relabeled.",
    "This loader performs no reset, recorder or dynamics replay; it returns the recorded bytes unchanged.",
)


def _read_bound(path, expected):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, f"Producer evidence changed: {Path(path).name}")
    return json.loads(raw)


def authenticate_producer():
    """The formal G2's sealed package and its attempt-1 record: every pinned source file, the PASS report, the
    collection identity of both worlds."""
    manifest = _read_bound(F2 / "G2_FORMAL_MANIFEST.json", MANIFEST_SHA)
    report = _read_bound(RUN / "report.json", REPORT_SHA)
    collection = _read_bound(RUN / "collection.json", COLLECTION_SHA)
    pins = {name: digest for section in ("files", "upstream") for name, digest in manifest[section].items()}
    require(len(pins) == 44, "Unexpected producer source closure")
    for name, expected in pins.items():
        require(sha_file(F2 / name) == expected, f"Producer source changed: {name}")
    for world, spec in ARCHIVES.items():
        require(collection.get(world) == {"path": spec["file"], "sha256": spec["content_sha256"], "episodes": 1280,
                                          "entropy": ENTROPY, "recorder_sha256": RECORDER_SHA,
                                          "behavior": "knee_s10_wide_window_v1"},
                f"Producer collection identity differs ({world})")
    require(report.get("collection") == collection and report.get("headline") == "PASS" and report.get("attempt") == 1
            and report.get("manifest_sha256") == MANIFEST_SHA
            and report.get("integrity_after") == {"manifest_sha256": MANIFEST_SHA, "ok": True},
            "Producer run record differs or did not pass")
    return {"manifest_sha256": MANIFEST_SHA, "report_sha256": REPORT_SHA, "collection_sha256": COLLECTION_SHA,
            "verified_source_files": len(pins),
            "validation_reused": "generate -> KneeDataset.__init__ -> validate(check_dynamics=True), before digest and save"}


def check_content(world, arrays, envelope):
    require(set(envelope) == {"metadata", "sha256"}, "Invalid manifest envelope")
    require(set(arrays) == set(D.DTYPES), "Array schema differs")
    metadata = json.loads(canonical_json(envelope["metadata"]))
    require(metadata.get("schema") == D.SCHEMA and metadata.get("seed") == ENTROPY and metadata.get("world") == world
            and metadata.get("provenance") == "research" and metadata.get("episodes") == 1280
            and metadata.get("horizon") == D.HORIZON and metadata.get("recorder_sha256") == RECORDER_SHA,
            f"Dataset metadata identity differs ({world})")
    for name, value in arrays.items():
        require(value.shape == SHAPES[name] and value.dtype.str == D.DTYPES[name], f"{name}: shape or dtype differs")
        require(value.dtype.kind != "f" or bool(np.all(np.isfinite(value))), f"{name}: non-finite value")
    datasets = load_datasets()
    immutable = {name: datasets.immutable_array(value) for name, value in arrays.items()}
    digest = datasets.content_digest(immutable, metadata)
    require(envelope["sha256"] == ARCHIVES[world]["content_sha256"] == digest, f"Canonical content digest differs ({world})")
    return immutable, metadata, digest


def load(path):
    """The authenticated dataset at `path`, which must be one of the two recorded archives; a KneeDataset."""
    path = Path(path).resolve()
    matches = [w for w, spec in ARCHIVES.items() if (RUN / spec["file"]).resolve() == path]
    require(len(matches) == 1, f"Only the two formal G2 v2 archives are accepted, not {path}")
    world = matches[0]
    provenance = authenticate_producer()
    require(sha_file(path) == ARCHIVES[world]["archive_sha256"], f"{world} archive bytes differ from the formal G2 record")
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == set(D.DTYPES) | {"manifest"}, "Unexpected NPZ members")
        envelope = json.loads(str(archive["manifest"]))
        arrays = {name: archive[name] for name in D.DTYPES}
    immutable, metadata, digest = check_content(world, arrays, envelope)
    result = object.__new__(D.KneeDataset)
    object.__setattr__(result, "_arrays", MappingProxyType(immutable))
    object.__setattr__(result, "_metadata_json", canonical_json(metadata))
    object.__setattr__(result, "_digest", digest)
    require(sha_file(path) == ARCHIVES[world]["archive_sha256"], "Archive changed while loading")
    return result


def receipt():
    return {"loader": "screen_data.load (authenticated reuse, no regeneration)", "archives": dict(ARCHIVES),
            "producer": authenticate_producer(), "resets": 0, "simulator_steps": 0, "limitations": list(LIMITATIONS)}


_ORIGINAL_LOAD = D.load


def patch():
    """Install change 7: `dataset_knee.load` becomes this loader for the two archives (anything else is refused)."""
    D.load = load
    return D

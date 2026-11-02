"""Immutable Path A learner data: original danger mask and reviewed slab cut.

No model, simulator, calibration fitting, or training is imported or executed.
The caller supplies the native Transition constructor. Initialization and later
execution remain subject to the separate training review and frozen runner.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
V2 = REPO / "experiments/thrusterchain_bca/two_scenario_v2"
REPAIR = REPO / "analysis/two_scenario_v2_repair_v1"
DATASET = V2 / "setup_attempt_1/parent_data.npz"
GEOMETRY = V2 / "setup_attempt_1/geometry.json"
DATASET_SHA256 = "0b9d51777803f2c0f5bf003f9b3fd0043427485856ebd33dcae4baecc184f426"
CONTENT_SHA256 = "194457ae13a172cb7e3af301b17f52abb8f17eb7a8c54cb27cef05064d2794ff"
GEOMETRY_SHA256 = "942271e7ed260356cfc4b4543fa79c585c0e2721bf19514bcbcfa581946c2411"
CALIBRATION_SEED = 121312521
PINS = {
    V2 / "MANIFEST.json": "c4e3198289e9ea8b329bbaedbf8cdc5e1c7cea19249beb121ffaa100af53336d",
    V2 / "protocol.json": "8df39bffe766b59c58033a96a82f18765085a636f5be7484814c6ef60686822a",
    GEOMETRY: GEOMETRY_SHA256,
    V2 / "setup_attempt_1/data_evidence.json": "125c1ccbde815fe5edca99724665f04b9197bc355e28f0f7a9d178f47a37c78f",
    REPAIR / "connector_cut.py": "c966571b8eb9feb17823e0b6f0859a9a9cfdfd4762d60ee030f853a208a0c107",
    REPAIR / "CUT_RULE.json": "d9b1be4ddd96fcaf4572442ec87eb0511e8973598fad9b867d3033e415fe8e4b",
    REPAIR / "CUT_REPORT.json": "3d2213ea3ada844e92b8ed645e8f1e0e2a20592cfc1b78cd22146d8bc6094d9d",
    REPAIR / "DANGER_CONTRACT.json": "49e83ef970ede060b8ed9d2210252a0fbaf239685d2c8b72cd5053e9a66a9dfe",
    REPAIR / "danger_attempt_1/RESULT.json": "654e2df43128f78a1035b592d0e58b2f2a5001966070cc5d4da51e043d4f220f",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_sources():
    for path, expected in PINS.items():
        if sha(path) != expected:
            raise ValueError("Path A dependency changed: " + str(path.relative_to(REPO)))
    manifest = read(V2 / "MANIFEST.json")
    for name in ("task.py", "data.py", "integration.py"):
        path = V2 / name
        if sha(path) != manifest["files"].get(path.relative_to(REPO).as_posix()):
            raise ValueError("Original v2 source differs from its seal: " + name)
    return {str(path.relative_to(REPO).as_posix()): value for path, value in PINS.items()}


SOURCE_PINS = verify_sources()
D = importlib.import_module("experiments.thrusterchain_bca.two_scenario_v2.data")
I = importlib.import_module("experiments.thrusterchain_bca.two_scenario_v2.integration")
C = importlib.import_module("analysis.two_scenario_v2_repair_v1.connector_cut")
for module, path in ((D, V2 / "data.py"), (D.T, V2 / "task.py"), (I, V2 / "integration.py"), (C, REPAIR / "connector_cut.py")):
    if Path(module.__file__).resolve() != path.resolve():
        raise ValueError("Path A module resolved outside its authenticated source")


def load_verified():
    verify_sources()
    records, envelope = D.load(DATASET, DATASET_SHA256)
    if envelope["content_sha256"] != CONTENT_SHA256:
        raise ValueError("Unexpected immutable dataset content")
    return SimpleNamespace(records=records, envelope=envelope, geometry=read(GEOMETRY),
                           protocol=read(V2 / "protocol.json"), cut_rule=read(REPAIR / "CUT_RULE.json"),
                           cut_report=read(REPAIR / "CUT_REPORT.json"),
                           original_data_evidence=read(V2 / "setup_attempt_1/data_evidence.json"))


def _validate_source(source):
    verify_sources()
    if D.digest(source.records, source.envelope["metadata"]) != CONTENT_SHA256 or source.envelope.get("content_sha256") != CONTENT_SHA256:
        raise ValueError("Raw dataset content or metadata was changed")
    if I.digest(source.geometry) != I.digest(read(GEOMETRY)):
        raise ValueError("Geometry content was changed")
    if source.protocol != read(V2 / "protocol.json") or source.cut_rule != read(REPAIR / "CUT_RULE.json") or source.cut_report != read(REPAIR / "CUT_REPORT.json") or source.original_data_evidence != read(V2 / "setup_attempt_1/data_evidence.json"):
        raise ValueError("Source protocol, mask rule or reviewed evidence was changed")
    if source.cut_report["decision"] != "saved_data_mask_checks_passed" or not all(source.cut_report["checks"].values()):
        raise ValueError("The reviewed separating-cut checks did not pass")
    danger = read(REPAIR / "danger_attempt_1/RESULT.json")
    if danger["decision"] != "passed_development_diagnostic" or not danger["prepared_identity_unchanged"]:
        raise ValueError("Required original-danger diagnostic did not pass")


def validate_protocol(protocol, original):
    I.validate_protocol(protocol)
    descriptive = {"schema", "status", "training_authorized", "scientific_evaluation_authorized"}
    for key, value in original.items():
        if key not in descriptive and protocol.get(key) != value:
            raise ValueError("Path A changed the frozen protocol or seeds: " + key)
    if protocol["calibration_seed"] != CALIBRATION_SEED or protocol["calibration_rows"] != 128:
        raise ValueError("Calibration selection must remain the original deterministic 128-parent rule")


def _array_hash(value, dtype=None):
    return hashlib.sha256(np.ascontiguousarray(value, dtype=dtype).tobytes()).hexdigest()


def reviewed_mask(source, condition):
    if condition not in I.CONDITIONS:
        raise ValueError("Unknown Path A data condition")
    a = source.records
    mask = D.retained_mask(a, source.geometry, "honeypot") if condition == "honeypot" else C.retained_mask(a, source.geometry)
    if mask.dtype != bool or mask.shape != (len(a["rewards"]),):
        raise ValueError("Invalid reviewed mask shape or dtype")
    row_hash = _array_hash(a["row_ids"][mask], "<i8")
    if condition == "honeypot":
        expected = source.original_data_evidence["honeypot"]["mask_row_ids_sha256"]
        if not source.original_data_evidence["honeypot"]["passed"] or row_hash != expected:
            raise ValueError("Honeypot retention changed from the original mask")
    else:
        expected = source.cut_report["hashes"]
        if row_hash != expected["retained_row_ids_int64le_sha256"] or _array_hash(mask, np.uint8) != expected["retained_mask_uint8_sha256"]:
            raise ValueError("Bottleneck retention differs from the reviewed separating slab")
    return I._frozen(mask, bool)


def prepare(native, condition, *, protocol=None, source=None):
    """Build actual read-only learner tuples; accepts no caller-provided mask.

    `native` supplies only Transition. Tests use a namedtuple rather than loading
    UniFloral/JAX. Both policy arms receive the same result for their condition.
    """
    source = load_verified() if source is None else source
    _validate_source(source)
    protocol = source.protocol if protocol is None else protocol
    validate_protocol(protocol, source.protocol)
    a = source.records
    mask = reviewed_mask(source, condition)
    roles, parents, row_ids = a["roles"], a["episode_ids"], a["row_ids"]
    train_rows = np.flatnonzero(mask & (roles == D.FIT))
    cal_rows = np.flatnonzero(mask & (roles == D.CALIBRATION))
    eligible = np.sort(np.unique(parents[cal_rows]))
    if len(train_rows) < 256 or len(eligible) < 128:
        raise ValueError("Insufficient retained fitting/calibration data; no backfill or mask relaxation")
    rng = np.random.Generator(np.random.PCG64(CALIBRATION_SEED))
    selected = np.asarray([int(rng.choice(cal_rows[parents[cal_rows] == parent])) for parent in eligible[:128]], np.int64)
    if np.intersect1d(parents[train_rows], parents[selected]).size or np.unique(parents[selected]).size != 128:
        raise ValueError("Calibration parent independence failed")

    def transition(indices):
        return native.Transition(*(I._frozen(a[key][indices], np.float32)
                                   for key in ("observations", "actions", "rewards", "next_observations", "terminals")))

    first = np.asarray([rows[np.argmin(row_ids[rows])] for parent in np.sort(np.unique(parents[train_rows]))
                        for rows in [train_rows[parents[train_rows] == parent]]], np.int64)
    fragments = np.full(len(mask), -1, np.int64)
    fragments[mask] = D.fragment_ids_for_mask(a, mask)
    train = transition(train_rows)
    review = source.cut_report
    metadata = {
        "binding": "two_scenario_path_a.v1", "condition": condition,
        "dataset_sha256": DATASET_SHA256, "dataset_content_sha256": CONTENT_SHA256,
        "geometry_sha256": GEOMETRY_SHA256, "geometry_content_sha256": I.digest(source.geometry),
        "protocol_sha256": I.digest(protocol), "source_pins": SOURCE_PINS,
        "mask_definition": "original v2 source/next/chord danger exclusion" if condition == "honeypot" else review["mask_definition"],
        "mask_rule_sha256": PINS[REPAIR / "CUT_RULE.json"] if condition == "bottleneck" else PINS[V2 / "MANIFEST.json"],
        "retention_mask_sha256": _array_hash(mask, np.uint8), "retained_row_ids_sha256": _array_hash(row_ids[mask], "<i8"),
        "training_row_ids_sha256": _array_hash(row_ids[train_rows], "<i8"),
        "training_tuple_fields_sha256": {key: _array_hash(a[key][train_rows], np.float32)
                                         for key in ("observations", "actions", "rewards", "next_observations", "terminals")},
        "training_rows": int(len(train_rows)), "training_parents": int(np.unique(parents[train_rows]).size),
        "training_fragment_ids_sha256": _array_hash(fragments[train_rows], "<i8"),
        "calibration_rows": 128, "calibration_parents": 128, "calibration_eligible_parents": int(len(eligible)),
        "calibration_seed": CALIBRATION_SEED,
        "calibration_rule": "first 128 eligible parents sorted by ID; one uniform retained original tuple per parent using PCG64",
        "calibration_row_ids_sha256": _array_hash(row_ids[selected], "<i8"),
        "calibration_parent_ids_sha256": _array_hash(parents[selected], "<i8"),
        "normalization": "all condition-retained fitting observations only; identical native/BCA pair",
        "normalization_source_row_ids_sha256": _array_hash(row_ids[train_rows], "<i8"),
        "normalization_observation_bytes_sha256": _array_hash(train.obs, np.float32),
        "next_action_used": False, "fragment_joining": False,
        "reward_and_done_precision": "original saved tuples cast to float32 exactly as the unchanged native Transition adapter",
        "reserved_parents_used_for_learning": 0,
        "calibration_claim": "No conditional missing-region or selected-action coverage guarantee",
        "freshness": "Previously inspected development data; no new confirmation claim",
        "training_review_required": True,
    }
    if condition == "bottleneck":
        metadata["reviewed_cut_evidence"] = {
            "report_sha256": PINS[REPAIR / "CUT_REPORT.json"],
            "intact_successful_fitting_trajectories": review["by_role"]["fitting"]["intact_retained_success_parents"],
            "genuine_fitting_fragment_pairs": review["by_role"]["fitting"]["genuine_fragment_pair_parents"],
            "B25_retained_fitting_parents": review["support"]["B"]["distinct_parents"],
            "C30_retained_fitting_parents": review["support"]["C"]["distinct_parents"],
        }
    return SimpleNamespace(train=train, heldout=transition(selected), probe=transition(first),
        scale_bank=transition(train_rows[:256]), normalization_fixture=train,
        source_row_ids=I._frozen(row_ids[train_rows], np.int64),
        source_episode_ids=I._frozen(parents[train_rows], np.int64),
        source_times=I._frozen(a["times"][train_rows], np.int32),
        source_fragment_ids=I._frozen(fragments[train_rows], np.int64),
        normalization_source_row_ids=I._frozen(row_ids[train_rows], np.int64),
        heldout_ids={"row_ids": I._frozen(row_ids[selected], np.int64), "episode_ids": I._frozen(parents[selected], np.int64)},
        metadata=metadata, receipt=metadata)

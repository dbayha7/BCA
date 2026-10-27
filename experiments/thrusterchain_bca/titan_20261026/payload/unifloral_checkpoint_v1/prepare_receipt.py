"""Stdlib-only AST, JSON, immutable-source and offline-file inspection."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
NAMES = ("training.py", "checkpoint_io.py", "export.py", "checks.py", "data.py", "run.py",
         "prepare_receipt.py", "config.json", "expected_results.json", "README.md")
files, syntax = {}, []
for name in NAMES:
    raw = (ROOT / name).read_bytes()
    files[name] = hashlib.sha256(raw).hexdigest()
    if name.endswith(".py"):
        compile(ast.parse(raw.decode("utf-8"), filename=name), name, "exec")
        syntax.append(name)
    elif name.endswith(".json"):
        json.loads(raw)
previous = {}
for folder, expected in (
    ("unifloral_td3bc_validation_v1", "1e26f713822020275179845f30c8691467951745d6c7c829ff017b7045904516"),
    ("unifloral_bca_pipeline_validation_v1", "6eb91c5bcc6c2edddc1ab93c19424307e12a3c8363bf9063fa4bbe2664180478"),
    ("unifloral_frozen_usefulness_v1", None)):
    base = ROOT.parent / folder
    raw = (base / "prepared_files.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected and digest != expected:
        raise ValueError("Previous validation manifest changed: " + folder)
    declared = json.loads(raw)["files"]
    for name, value in declared.items():
        if hashlib.sha256((base / name).read_bytes()).hexdigest() != value:
            raise ValueError("Previous source changed: " + name)
    previous[folder] = {"manifest_sha256": digest, "verified_files": len(declared)}
native_root = ROOT.parent / "unifloral_td3bc_validation_v1/upstream"
native_manifest = json.loads((ROOT.parent / "unifloral_td3bc_validation_v1/provenance/unifloral.json").read_bytes())
for name, digest in native_manifest["source_sha256"].items():
    if hashlib.sha256((native_root / name).read_bytes()).hexdigest() != digest:
        raise ValueError("Native pinned source changed: " + name)
data_path = Path(r"C:\Users\David Bayha\Downloads\titan_transfer_20261022_v1\titan_transfer_20261022_v1\research_runs\pilot_20261023T115408Z\regular_1280_2026102301.npz")
data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
if data_hash != "4c857aaa0373bee4c74db33036303b4cf75c703b289a459525f75090741f3226":
    raise ValueError("Pinned offline dataset changed")


def read_npy(archive, name):
    raw = archive.read(name + ".npy")
    if raw[:6] != b"\x93NUMPY":
        raise ValueError("Unexpected NPY member")
    major = raw[6]
    offset = 10 if major == 1 else 12
    size = struct.unpack("<H" if major == 1 else "<I", raw[8:offset])[0]
    header = ast.literal_eval(raw[offset:offset+size].decode("latin1"))
    if header["fortran_order"]:
        raise ValueError("Unexpected array order")
    return header, raw[offset+size:]


with zipfile.ZipFile(data_path) as archive:
    header, payload = read_npy(archive, "manifest")
    if header["shape"] != () or not header["descr"].startswith("<U"):
        raise ValueError("Unexpected dataset manifest dtype")
    envelope = json.loads(payload.decode("utf-32le"))
    vectors = {}
    for name, expected_dtype, fmt in (("group_ids", "<i8", "<q"),
                                     ("episode_seeds", "<i8", "<q"),
                                     ("split_codes", "|u1", "B")):
        header, payload = read_npy(archive, name)
        if header["shape"] != (1280,) or header["descr"] != expected_dtype:
            raise ValueError("Unexpected offline provenance array")
        vectors[name] = [value[0] for value in struct.iter_unpack(fmt, payload)]
metadata = envelope["metadata"]
if envelope["sha256"] != "6b58db68263dff0d7e9b8d6dfa7c1ca5d4127ed9ee6a3b9069add1c565e88331":
    raise ValueError("Dataset content digest differs")
train_indices = [i for i, code in enumerate(vectors["split_codes"]) if code == 0]
cal_pool = [i for i, code in enumerate(vectors["split_codes"]) if code == 1]
cal_indices = sorted(cal_pool, key=lambda i: vectors["group_ids"][i])[:128]
reserved = set(range(2026102400, 2026102464)) | set(range(2026102500, 2026102564))
if len(train_indices) != 768 or len(cal_pool) != 256 or len(cal_indices) != 128:
    raise ValueError("Dataset split count mismatch")
if any(vectors["episode_seeds"][i] in reserved for i in train_indices + cal_indices):
    raise ValueError("Dataset/evaluation episode seed overlap")
receipt = {"utc": datetime.now(timezone.utc).isoformat(), "status": "static_checks_passed_runtime_not_executed",
    "python": sys.version, "syntax_checked": syntax, "files": files, "previous_sources": previous,
    "native_original_files_verified": len(native_manifest["source_sha256"]),
    "dataset_inspection": {"path": str(data_path), "file_sha256": data_hash,
        "content_digest": envelope["sha256"], "split_counts": metadata["split_counts"],
        "selected_calibration_groups": 128, "reserved_episode_seed_overlap": False,
        "scope": "Stdlib ZIP/NPY-header/provenance inspection; no simulator or NumPy execution"},
    "scientific_imports": 0, "optimizer_updates": 0, "simulator_transitions": 0,
    "runtime_gate_status": "pending_user_GPU_execution", "full_training_started": False}
for name, value in (("prepared_files.json", {"schema": "prepared_files.v1", "files": files}),
                    ("PREPARATION_RECEIPT.json", receipt)):
    with (ROOT / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
print(json.dumps({"status": "static_checks_passed", "python_files": len(syntax),
                  "native_files": len(native_manifest["source_sha256"]), "dataset_sha256": data_hash,
                  "runtime_executed": False}))

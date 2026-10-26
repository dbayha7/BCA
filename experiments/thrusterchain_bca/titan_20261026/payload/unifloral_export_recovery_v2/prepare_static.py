"""One-time standard-library regression and provenance preparation for v2."""
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from integrity import content_identity, mapping_order_audit

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "unifloral_export_recovery_v1"
ORIGINAL = ROOT.parent / "unifloral_checkpoint_v1"
RUN = ORIGINAL / "train_20261023T183248_552877Z"
EVALUATOR = ROOT.parent / "unifloral_frozen_usefulness_v1_seedfix"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    for name in ("REPAIR_RECEIPT.json", "STATIC_CHECKS.json", "prepared_files.json"):
        if (ROOT / name).exists():
            raise FileExistsError(str(ROOT / name))
    prior = read(PREVIOUS / "REPAIR_RECEIPT.json")
    preserved = dict(prior["preserved_files"])
    for name, expected in read(PREVIOUS / "prepared_files.json")["files"].items():
        path = PREVIOUS / name
        if sha(path) != expected:
            raise ValueError("Prior recovery source changed: " + name)
        preserved[path.relative_to(ROOT.parent).as_posix()] = expected
    for name in ("prepared_files.json", "recovery_20261023T203148_216211Z/report.json",
                 "recovery_20261023T203148_216211Z/worker.json"):
        path = PREVIOUS / name
        preserved[path.relative_to(ROOT.parent).as_posix()] = sha(path)
    for relative, expected in preserved.items():
        if sha(ROOT.parent / relative) != expected:
            raise ValueError("Preserved source/checkpoint/evidence changed: " + relative)
    if (ROOT / "export_seedfix.py").read_bytes() != (PREVIOUS / "export_seedfix.py").read_bytes():
        raise ValueError("Exporter changed beyond the prior seed fix")
    if sha(EVALUATOR / "prepared_files.json") != prior["evaluator_manifest_sha256"]:
        raise ValueError("Evaluator manifest changed")
    for name, expected in read(EVALUATOR / "prepared_files.json")["files"].items():
        if sha(EVALUATOR / name) != expected:
            raise ValueError("Evaluator changed: " + name)
    meta = read(RUN / "checkpoints/critic_020000_terminal/metadata.json")
    expected = meta["state_leaves"]
    baseline = content_identity(expected, expected)
    if baseline["numeric_leaves"] != 215:
        raise AssertionError("Expected 215 numeric leaves")
    regressions = [{"name": "complete_215_leaf_description_accepted", "pass": True}]

    def first_leaf(value):
        if isinstance(value, dict):
            if set(value) == {"dtype", "shape", "sha256"}:
                return value
            for child in value.values():
                found = first_leaf(child)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = first_leaf(child)
                if found is not None:
                    return found
        return None

    for label, key, value in (("value_change_rejected", "sha256", "0" * 64),
                              ("dtype_change_rejected", "dtype", "<i8"),
                              ("shape_change_rejected", "shape", [1])):
        altered = copy.deepcopy(expected)
        first_leaf(altered)[key] = value
        try:
            content_identity(altered, expected)
        except ValueError:
            regressions.append({"name": label, "pass": True})
        else:
            raise AssertionError(label)
    for label, altered in (("missing_state_path_rejected", {"dict": expected["dict"][:-1]}),
                            ("additional_state_path_rejected", {"dict": expected["dict"] + [["unexpected", {"none": True}]]})):
        try:
            content_identity(altered, expected)
        except ValueError:
            regressions.append({"name": label, "pass": True})
        else:
            raise AssertionError(label)
    reordered = copy.deepcopy(expected)
    leaf = first_leaf(reordered)
    reverse = list(reversed(list(leaf.items())))
    leaf.clear()
    leaf.update(reverse)
    if content_identity(reordered, expected) != baseline:
        raise AssertionError("Descriptor mapping order affects content identity")
    regressions.append({"name": "descriptor_key_order_preserves_exact_identity", "pass": True})
    audit = mapping_order_audit({"params": {"bias": 0, "kernel": 1}}, {"params": {"kernel": 1, "bias": 0}})
    if audit["changed_mapping_order_count"] != 1:
        raise AssertionError("Mapping order audit failed")
    regressions.append({"name": "map_order_difference_is_recorded", "pass": True})
    # Re-run the actual unchanged exporter seed rejection expression, not a duplicate implementation.
    tree = ast.parse((ROOT / "export_seedfix.py").read_bytes())
    tests = [node.test for node in ast.walk(tree) if isinstance(node, ast.If)
             and "used_seeds" in ast.unparse(node.test) and "2 ** 63" in ast.unparse(node.test)]
    if len(tests) != 1:
        raise AssertionError("Exporter seed guard not found")
    code = compile(ast.Expression(tests[0]), "actual_export_seed_guard", "eval")
    info = read(RUN / "data_split_receipt.json")
    seeds = info["used_episode_seeds"]
    fixtures = [("actual_valid_seeds", seeds, False), ("missing", None, True),
                ("empty", [], True), ("wrong_count", seeds[:-1], True)]
    for label, value, rejected in (("zero", 0, False), ("uint32_boundary", 2**32, False),
            ("int63_max", 2**63-1, False), ("negative", -1, True), ("overflow", 2**63, True),
            ("float", 1.5, True), ("bool", True, True), ("str", "1", True),
            ("duplicate", seeds[1], True), ("evaluation_overlap", 2026102400, True),
            ("confirmation_overlap", 2026102500, True)):
        fixtures.append((label, [value] + seeds[1:], rejected))
    for label, values, wanted in fixtures:
        result = eval(code, {"__builtins__": {"isinstance": isinstance, "list": list, "len": len,
                      "set": set, "any": any, "type": type, "int": int, "range": range}},
                      {"used_seeds": values, "train_groups": info["train_groups"], "calibration_groups": info["calibration_groups"]})
        if bool(result) != wanted:
            raise AssertionError("Seed fixture: " + label)
    files = sorted(ROOT.glob("*.py"))
    for path in files:
        ast.parse(path.read_bytes(), filename=str(path))
    worker = ast.parse((ROOT / "recover.py").read_bytes())
    forbidden = [ast.unparse(node.func) for node in ast.walk(worker) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr in ("step", "fit_scale", "refresh", "train", "reset")]
    if forbidden:
        raise AssertionError("Unexpected scientific update or simulator call: " + repr(forbidden))
    # The five parity checks retain their existing numeric tolerance.
    if 'np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6)' not in (ROOT / "recover.py").read_text():
        raise AssertionError("Numerical tolerance changed")
    receipt = {**prior, "schema": "unifloral.export_recovery.v2", "utc": datetime.now(timezone.utc).isoformat(),
        "preserved_files": preserved,
        "prior_failed_recovery": "unifloral_export_recovery_v1/recovery_20261023T203148_216211Z/report.json",
        "new_recovery_change": "Separate immutable original payload SHA, exact numeric-content identity and runtime reencoding SHA; preserve every numerical check and tolerance",
        "flax_source_reference": "https://raw.githubusercontent.com/google/flax/v0.10.7/flax/serialization.py",
        "flax_mechanism": "Dictionary restoration follows the template's iteration order; to_bytes preserves current dictionary order",
        "observed_wire_audit": {"numeric_leaves": 215, "exact_metadata_content_match": True,
                                "saved_dense_parameter_order": ["bias", "kernel"],
                                "actual_restored_order": "not captured in v1; v2 will record"},
        "exporter_unchanged_from_v1": True, "evaluator_unchanged_from_seedfix": True,
        "runtime_executed": False, "canonical_expected_state_identity": baseline}
    write(ROOT / "REPAIR_RECEIPT.json", receipt)
    write(ROOT / "STATIC_CHECKS.json", {"status": "pass", "python_asts": [path.name for path in files],
          "exact_content_regressions": regressions, "seed_guard_regressions": len(fixtures),
          "numerical_tolerances_unchanged": True, "original_evidence_hashes_unchanged": True,
          "scientific_imports": False, "runtime_recovery": "pending_user_execution"})
    names = [p.name for p in files] + ["README.md", "REPAIR_RECEIPT.json", "STATIC_CHECKS.json"]
    write(ROOT / "prepared_files.json", {"schema": "prepared_files.v1", "files": {name: sha(ROOT / name) for name in names}})
    print(json.dumps({"status": "static_pass", "content_checks": len(regressions),
                      "seed_checks": len(fixtures), "manifest_sha256": sha(ROOT / "prepared_files.json")}))


if __name__ == "__main__":
    main()

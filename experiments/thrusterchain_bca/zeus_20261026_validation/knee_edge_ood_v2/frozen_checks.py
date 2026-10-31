"""Known-answer reproduction on the frozen central-quartile native 20k cells (protocol 4.8; step 0.3, user-run).

Agents never open checkpoints. This module holds the pure comparison rules and the preflight; the user-run step 0.3
driver (prepared and dry-run-reviewed with the A2 package) builds each policy through rerun1's own path and calls
these functions. Reference: central_quartile_training_v1_evaluation_rerun1/run/evaluation/*__native__20000__*.json.

Rules:
- slots 1-5: the single probe builder's FIXED_A0 arrays equal saved actions[:, 1:, :] exactly;
- slot 0: the actor's unclipped command, compared as a full 4-vector at |d| <= 1e-6 x max(1, |value|);
- q1, twin_min, adjusted on all six slots (slot 0 fed the saved actions[:, 0]) at the same relative tolerance; U is
  (q1 - adjusted) / 3;
- saved lambda to 1e-6 relative;
- saved per-probe Q^pi (32 x 6) to 1e-9 relative on all six slots, with the recorded batching and execution stack;
- preflight: runtime_receipt and the device kind are asserted once before any value is computed or written; a
  preflight failure with zero values in the ledger is the narrow exception to 5.0's second-failure rule.
"""
import json
from pathlib import Path

import numpy as np

from core import RERUN1, VALIDATION, require, sha_file
from probes import FIXED_A0, known_answer_slots_1_to_5

REFERENCE_GLOB = "run/evaluation/*__native__20000__*.json"
RECORDED_DEVICE_KIND = "NVIDIA GeForce RTX 5070 Ti Laptop GPU"
Q_RTOL, QPI_RTOL, LAMBDA_RTOL = 1e-6, 1e-9, 1e-6


def reference_files():
    files = sorted((RERUN1).glob(REFERENCE_GLOB))
    require(len(files) == 18, f"Expected 18 native 20k reference files (9 cells x 2 contexts), found {len(files)}")
    return files


def read_reference(path):
    """The saved per-probe arrays of one (cell, context) file. Reads JSON only."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    probe = record["q"]["per_probe"]
    out = {"actions": np.asarray(probe["actions"], np.float32), "q1": np.asarray(probe["q1"], np.float64),
           "qpi": np.asarray(probe["qpi"], np.float64),
           "twin_min": None if probe.get("twin_min") is None else np.asarray(probe["twin_min"], np.float64),
           "adjusted": None if probe.get("adjusted") is None else np.asarray(probe["adjusted"], np.float64),
           "sha256": sha_file(path), "file": Path(path).name}
    require(out["actions"].shape == (32, 6, 4) and out["q1"].shape == (32, 6) and out["qpi"].shape == (32, 6),
            "Saved per-probe arrays must be (32, 6, 4) and (32, 6)")
    return out


def relative_ok(new, saved, rtol, floor=1.0):
    """|new - saved| <= rtol x max(floor, |saved|). Q values use floor 1 (4.8: 1e-6 x max(1, |value|)); lambda is
    compared purely relatively (floor 0), because at about 0.037 a floor of 1 would loosen it about 27-fold."""
    new, saved = np.asarray(new, np.float64), np.asarray(saved, np.float64)
    require(new.shape == saved.shape, "Shapes differ")
    scale = np.maximum(floor, np.abs(saved)) if floor > 0 else np.abs(saved)
    require(bool(np.all(scale > 0)), "Pure relative comparison needs nonzero saved values")
    scaled = np.abs(new - saved) / scale
    return bool(np.all(scaled <= rtol)), float(scaled.max())


CELLS = tuple(f"seed{s}_{c}" for s in range(3) for c in ("full", "hole", "random"))
LAMBDA_SOURCES = ("central_quartile_training_v1_authorized1/run", "central_quartile_training_v1_continuation1/run")


def saved_lambdas(cell):
    """The cell's saved native 20k lambdas, one per gradient-audit batch (lambda_from_unpenalized_Q1), from exactly
    one of authorized1 or continuation1. Reads JSON only."""
    require(cell in CELLS, "Unknown cell")
    found = [VALIDATION / root / cell / "native/gradient_audit_020000.json" for root in LAMBDA_SOURCES]
    found = [p for p in found if p.is_file()]
    require(len(found) == 1, f"{cell}: expected exactly one native 20k gradient audit, found {len(found)}")
    record = json.loads(found[0].read_text(encoding="utf-8"))
    values = np.asarray([b["lambda_from_unpenalized_Q1"] for b in record["batches"]], np.float64)
    require(len(values) > 0 and bool(np.all(np.isfinite(values))), f"{cell}: no saved lambda")
    return {"values": values, "file": str(found[0].relative_to(VALIDATION)), "sha256": sha_file(found[0])}


def check_slots_1_to_5(saved_actions):
    built = known_answer_slots_1_to_5(len(saved_actions))
    return bool(np.array_equal(built, np.asarray(saved_actions)[:, 1:, :])), {"fixed_a0": list(FIXED_A0)}


def check_cell(saved, *, slot0, q1, twin_min, adjusted, qpi, lam, saved_lambda):
    """All 4.8 per-cell comparisons; every argument is required. `lam` and `saved_lambda` are per-batch arrays."""
    require(saved["twin_min"] is not None and saved["adjusted"] is not None, "Saved twin_min and adjusted are required")
    require(lam is not None and saved_lambda is not None, "The 4.8 lambda comparison is required")
    results = {"slots_1_to_5_exact": check_slots_1_to_5(saved["actions"])[0]}
    results["slot0"] = relative_ok(slot0, saved["actions"][:, 0], Q_RTOL)
    results["q1"] = relative_ok(q1, saved["q1"], Q_RTOL)
    results["twin_min"] = relative_ok(twin_min, saved["twin_min"], Q_RTOL)
    results["adjusted"] = relative_ok(adjusted, saved["adjusted"], Q_RTOL)
    results["qpi"] = relative_ok(qpi, saved["qpi"], QPI_RTOL)
    results["lambda"] = relative_ok(lam, saved_lambda, LAMBDA_RTOL, floor=0.0)
    ok = all(v if isinstance(v, bool) else v[0] for v in results.values())
    return ok, results


def width_from_adjusted(q1, adjusted):
    return (np.asarray(q1, np.float64) - np.asarray(adjusted, np.float64)) / 3.0


def preflight(receipt_function, native, ledger):
    """Assert the recorded runtime receipt and device kind before any reproduction value exists.

    Returns ("PASS", receipt) or ("INCOMPLETE_PREFLIGHT", reason). The exemption from 5.0's second-failure rule
    applies only when the ledger shows zero reproduction values for the attempt.
    """
    require(ledger.get("reproduction_values", 0) == 0, "Preflight must run before any reproduction value")
    try:
        receipt = receipt_function(native)
        require(receipt["device_kinds"] and all(k == RECORDED_DEVICE_KIND for k in receipt["device_kinds"]),
                "Device kind differs from the recorded one")
    except Exception as exc:          # wrong venv (PackageNotFoundError), missing attribute, missing key, ...
        ledger["preflight_failures"] = ledger.get("preflight_failures", 0) + 1
        return "INCOMPLETE_PREFLIGHT", f"{type(exc).__name__}: {exc}"
    return "PASS", receipt


def failure_counts_under_5_0(label, ledger):
    """True when a failure counts toward 5.0's second-failure rule (everything except a zero-value preflight)."""
    return not (label == "INCOMPLETE_PREFLIGHT" and ledger.get("reproduction_values", 0) == 0)

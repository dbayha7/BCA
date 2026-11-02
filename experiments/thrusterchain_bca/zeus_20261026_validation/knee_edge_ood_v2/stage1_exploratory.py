"""EXPLORATORY Stage 1 run of knee_edge_ood_v2, OUTSIDE the protocol. Not A4.

Step 0.4 (A3) ended INCONCLUSIVE (L2 item 5: 42 of 48 anchors predictor-valid in continuation seeds 1 and 2, bar 44),
and the protocol's 5.0 rules stop the gated line there. The user chose to run Stage 1 anyway, as an exploratory study.
This driver runs `stage1.execute` unchanged (same workers, data, seeds, caps, decision rules and 5.0 attempt logic), with
exactly these differences, all recorded:
- the step 0.4 gate accepts any completed step 0.4 attempt and RECORDS its label and reasons instead of requiring PASS;
- the approval file is EXPLORATORY_APPROVAL.json, and its words say the run is outside the protocol and supports no claim;
- output goes to stage1_exploratory_run/, never stage1_run/, so Stage 2 (A5) cannot be requested on it;
- every report carries an EXPLORATORY block naming the step 0.4 label this run ignores.
The decision rules are computed and recorded as usual so the outcome is visible, but no label from this run is a
protocol PASS, FAIL or INCONCLUSIVE: it is exploratory, and it is disclosed as such wherever it is cited.
"""
from pathlib import Path
import argparse
import json
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import attempts as A
import stage1
import step03
import step04
from core import require, sha_file

RUN = HERE / "stage1_exploratory_run"
APPROVAL = HERE / "EXPLORATORY_APPROVAL.json"
APPROVAL_WORDS = ("I approve an EXPLORATORY Stage 1 run of knee_edge_ood_v2 outside the protocol: step 0.4 ended "
                  "INCONCLUSIVE, so this run is not A4 and supports no claim")
DISCLOSURE = ("EXPLORATORY: knee_edge_ood_v2 step 0.4 (A3) ended INCONCLUSIVE and the gated protocol stopped there. "
              "This Stage 1 run ignores that stop by the user's decision. It is not A4; its outcome is not a protocol "
              "label and supports no claim about the knee-edge OOD hypothesis.")


def a3_outcome_exploratory(approval):
    """Any completed step 0.4 attempt; its label and reasons are recorded, not required."""
    attempts = A.history(step04.RUN)
    require(attempts and (attempts[-1] / "report.json").is_file(), "No step 0.4 attempt to run on")
    last = json.loads((attempts[-1] / "report.json").read_text(encoding="utf-8"))
    require(last.get("approval", {}).get("manifest_sha256") == stage1.A3_MANIFEST_SHA256,
            "The step 0.4 report was not produced under the committed A3 package")
    require(stage1.PREDICTION_ACK in approval.get("approval", ""), "The approval must acknowledge the standing prediction")
    sealed = last["approval"].get("sealed_units", {})
    collection_dir = step04.RUN / sealed["collection"] if "collection" in sealed else attempts[-1]
    record = json.loads((collection_dir / "collection.json").read_text(encoding="utf-8"))
    require(record["unit_label"] == "COMPLETE", "The collection unit is not complete")
    return {"label": last.get("label"), "report": str(attempts[-1] / "report.json"),
            "collection_dir": str(collection_dir), "collection_sha256": sha_file(collection_dir / "collection.json"),
            "EXPLORATORY": {"disclosure": DISCLOSURE, "step04_label_ignored": last.get("label"),
                            "step04_reasons": last.get("reasons"), "step04_l2": last.get("l2"),
                            "driver_sha256": sha_file(HERE / "stage1_exploratory.py")}}


def _finish_exploratory(directory, report):
    report["EXPLORATORY"] = report.get("a3", {}).get("EXPLORATORY", {"disclosure": DISCLOSURE})
    report["protocol_label"] = None                     # the computed label is exploratory, not a protocol outcome
    report["exploratory_label"] = report.get("label")
    return stage1_finish(directory, report)


stage1_finish = stage1._finish


def execute(dispatch=None):
    stage1.RUN, stage1.APPROVAL, stage1.APPROVAL_WORDS = RUN, APPROVAL, APPROVAL_WORDS
    stage1.a3_outcome, stage1._finish = a3_outcome_exploratory, _finish_exploratory
    RUN.mkdir(parents=True, exist_ok=True)
    marker = RUN / "EXPLORATORY.json"
    if not marker.exists():
        step03._write_new(marker, {"disclosure": DISCLOSURE, "approval_file": APPROVAL.name, "approval_words": APPROVAL_WORDS})
    print(json.dumps({"EXPLORATORY": DISCLOSURE}, indent=2))
    return stage1.execute(dispatch=dispatch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    result = execute()
    print(json.dumps({"EXPLORATORY": True, "exploratory_label": result.get("label"), "reasons": result.get("reasons"),
                      "reading": result.get("reading"), "step04_label_ignored": result.get("a3", {}).get("label")},
                     indent=2, default=step03._plain))

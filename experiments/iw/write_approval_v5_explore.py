"""Add the owner's exploratory go for the BCA training arms to runs/shift_iw_v5/approval.json (orchestrator only).

The pre-registered report allows no BCA arm on any quality (bca_allowed and bca_lower_edge_allowed are false
everywhere), so the official stages keep refusing. The owner asked, in chat on 2026-10-13, to run the arms anyway,
labelled exploratory. The exploratory runner, which is separate from the package and writes to
runs/shift_iw_v5_explore/, checks these scopes. The pre-registered report and claims are not changed.
"""
import hashlib
import json
from pathlib import Path

P = Path("/root/bca-shift/runs/shift_iw_v5/approval.json")
QS = ["random", "medium-replay", "medium", "medium-expert", "expert"]

body = json.loads(P.read_text())
assert body["kind"] == "shift-iw-approval" and body["scope"] == ["band", "coverage"]
body["scope"] = ["band", "coverage", "bca", "bca_lower_edge"]
body["qualities"] = {"bca": QS, "bca_lower_edge": QS}
body["exploratory_go"] = {
    "when": "2026-10-13, chat, after the version-5 report",
    "earlier_go_verbatim": ("Message: \"okay whats the status on bca and iw bca? we need them run if environment is "
                            "working correctly\". Question: \"When the report confirms the environment works, should I "
                            "train the BCA arms (unweighted vs importance-weighted) automatically on every quality where "
                            "their checks pass?\" Answer: \"Yes, both kinds, all qualities (Recommended)\"."),
    "question": "No quality qualified for the BCA training arms under the pre-registered gates. What next?",
    "answer_verbatim": "Run the arms anyway, labelled exploratory (Recommended)",
    "option_text": ("About 20 minutes. Trains unweighted vs weighted BCA (both uses of the band) on all five qualities, "
                    "clearly marked as outside the pre-registered claims, so we see the actual numbers rather than "
                    "predictions."),
    "status": "EXPLORATORY: outside the pre-registered claims; the version-5 report's gates did not pass on any quality",
}
text = json.dumps(body, indent=1, sort_keys=True) + "\n"
P.write_text(text)
(P.parent / "approval.json.sha256").write_text(f"{hashlib.sha256(text.encode()).hexdigest()}  approval.json\n")
print("updated", P, hashlib.sha256(text.encode()).hexdigest())

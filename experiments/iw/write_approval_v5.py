"""Write runs/shift_iw_v5/approval.json from the approval template, with the owner's chat approval verbatim.

Written by the orchestrator only, from the owner's genuine answer in the chat on 2026-10-13.
"""
import hashlib
import json
from pathlib import Path

TEMPLATE = Path("/tmp/approval_template.json")
OUT_DIR = Path("/root/bca-shift/runs/shift_iw_v5")

QUESTION = ("Approve version 5 (freeze the spec, then data, TD3+BC, room checks, BCA's band and coverage for all five "
            "qualities), with confirmations (a) and (b) as described?")
ANSWER = "Yes, freeze and run"
CONTEXT = ("(a) when a state's band cannot be certified, the lower edge uses the largest finite threshold of its table; "
           "(b) in the lower edge's room check, where the critic alone never prefers a throttle past 80%, the "
           "'pulls the best command back below 80%' part is recorded as not applicable.")

body = json.loads(TEMPLATE.read_text())
assert body["kind"] == "shift-iw-approval" and body["scope"] == ["band", "coverage"] and body["qualities"] == {}
body["user_approval_verbatim"] = f"Question: {QUESTION} Answer: {ANSWER}"
body["approval_context"] = CONTEXT
body["approved_in"] = "chat answer, 2026-10-13 (owner's chosen approval mode for this experiment)"
OUT_DIR.mkdir(parents=True, exist_ok=True)
path = OUT_DIR / "approval.json"
text = json.dumps(body, indent=1, sort_keys=True) + "\n"
path.write_text(text)
(OUT_DIR / "approval.json.sha256").write_text(f"{hashlib.sha256(text.encode()).hexdigest()}  approval.json\n")
print("wrote", path, hashlib.sha256(text.encode()).hexdigest())

"""READ-ONLY: can the saved plain TD3+BC host snapshot (stalling world, gapped log, alpha 5) be loaded with the
learners' own template, and does it hold critic, critic target, actor and actor target? No outcome is read."""
import json, sys, hashlib
sys.path.insert(0, "/home/user/bca")
import experiments.shift.learners as L  # noqa: E402
d = "/home/user/bca/runs/shift/68d9e8779e148b7b/hosts/base/stall/gap/alpha5/plain/seed0"
rec = json.load(open(f"{d}/record.json"))
print(list(rec.keys())[:12])
res = rec.get("result", rec)
print({k: res[k] for k in list(res.keys())[:8] if not isinstance(res[k], (list, dict))})
snaps = res.get("snapshots") or rec.get("snapshots")
print(snaps[-1] if snaps else None)
blob = open(f"{d}/snapshot_100.msgpack", "rb").read()
print("sha", hashlib.sha256(blob).hexdigest()[:16])
st = L.load_tree(f"{d}/snapshot_100.msgpack", L.agent_template())
print(type(st).__name__, [f for f in st._fields] if hasattr(st, "_fields") else None)

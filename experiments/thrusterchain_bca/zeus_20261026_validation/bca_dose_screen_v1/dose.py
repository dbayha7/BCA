"""kappa_A for the BCA dose screen (PROTOCOL.md, "Dose rule"): the median over the 12 exploratory forks of each fork's
median dose needed at its 10k warmup gradient audit, rounded to the nearest integer. JSON only.

Also recorded per fork, all from the 10k fork audits and no branch outcome: the scored and outward row counts; the
share of outward rows where BCA points the wrong way, which kappa_A amplifies outward; the predicted reversal share at
kappa_A; and the share of all 768 probe rows where the penalty's 4-D action gradient at kappa_A outweighs native's.

  .venv/Scripts/python.exe -B bca_dose_screen_v1/dose.py
"""
from pathlib import Path
import hashlib
import json

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "knee_edge_ood_v2" / "stage1_exploratory_run" / "attempt_1"
KNEE = float(np.float32(0.8))
KAPPA_AS_BUILT = 3.0
FORKS = [f"s{s}_{a}_{w}" for s in (0, 1, 2) for a in ("cutoff", "random") for w in ("stall3", "nostall")]


def _rows(audit_path):
    rows = json.loads(Path(audit_path).read_text(encoding="utf-8"))["per_row"]
    a0 = np.asarray(rows["actor_actions"], np.float64)[:, 0]
    high = np.asarray(rows["high_load"], bool)
    g = {k: np.asarray(v, np.float64) for k, v in rows["action_loss_gradients"].items()}
    return a0, high, g


def fork_dose(audit_path):
    a0, high, g = _rows(audit_path)
    native, bca = g["q"][:, 0] + g["bc"][:, 0], g["prospective_bca3"][:, 0]
    use = high & (a0 < -KNEE) & (native > 0) & (bca < 0)
    dose = KAPPA_AS_BUILT * native[use] / (-bca[use])
    return float(np.median(dose)), int(use.sum())


def fork_disclosure(audit_path, kappa):
    a0, high, g = _rows(audit_path)
    native, bca = g["q"][:, 0] + g["bc"][:, 0], g["prospective_bca3"][:, 0]
    scored = high & (a0 < -KNEE)
    outward = scored & (native > 0)
    scale = kappa / KAPPA_AS_BUILT
    n4 = np.linalg.norm(g["q"] + g["bc"], axis=1)
    p4 = np.linalg.norm(scale * g["prospective_bca3"], axis=1)
    return {"scored_rows": int(scored.sum()), "outward_rows": int(outward.sum()),
            "wrong_direction_share_of_outward": float(np.mean(bca[outward] >= 0)) if outward.any() else None,
            "predicted_reversal_share_at_kappa_A": float(np.mean(native[outward] + scale * bca[outward] < 0)) if outward.any() else None,
            "penalty_outweighs_native_share_all_rows": float(np.mean(p4 > n4)),
            "median_penalty_to_native_norm_ratio_all_rows": float(np.median(p4 / np.maximum(n4, 1e-12)))}


def compute():
    per_fork = {}
    for name in FORKS:
        path = RUN / f"warmup_{name}" / "gradient_audit_010000.json"
        dose, n = fork_dose(path)
        per_fork[name] = {"median_dose": dose, "rows": n, "audit_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    kappa_unrounded = float(np.median([v["median_dose"] for v in per_fork.values()]))
    kappa = int(round(kappa_unrounded))
    for name in FORKS:
        per_fork[name].update(fork_disclosure(RUN / f"warmup_{name}" / "gradient_audit_010000.json", kappa))
    doses = [v["median_dose"] for v in per_fork.values()]
    return {"kappa_A": kappa, "kappa_A_unrounded": kappa_unrounded, "per_fork": per_fork,
            "per_fork_dose_range": [float(min(doses)), float(max(doses))],
            "rule": "PROTOCOL.md Dose rule", "protocol_sha256": hashlib.sha256((HERE / "PROTOCOL.md").read_bytes()).hexdigest()}


if __name__ == "__main__":
    out = compute()
    (HERE / "DOSE.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("kappa_A", "kappa_A_unrounded", "per_fork_dose_range")}))
    for name, v in out["per_fork"].items():
        print(name, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items() if k != "audit_sha256"})

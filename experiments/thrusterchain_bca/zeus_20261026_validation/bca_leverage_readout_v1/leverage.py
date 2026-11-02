"""BCA leverage readout v1 (rule: RULE.md, written first). JSON only: the saved gradient audits of the knee-edge
exploratory Stage 1. No checkpoint, no training. Exploratory.

  .venv/Scripts/python.exe -B bca_leverage_readout_v1/leverage.py
"""
from pathlib import Path
import hashlib
import json

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "knee_edge_ood_v2" / "stage1_exploratory_run" / "attempt_1"
KNEE = float(np.float32(0.8))
KAPPA, KAPPA_MAX = 3.0, 10.0
CHECKPOINTS = {"010000": "warmup", "015000": "branch", "020000": "branch"}


def readout(path):
    g = json.loads(path.read_text(encoding="utf-8"))
    rows = g["per_row"]
    a = np.asarray(rows["actor_actions"], np.float64)
    high = np.asarray(rows["high_load"], bool)
    q = np.asarray(rows["action_loss_gradients"]["q"], np.float64)[:, 0]
    bc = np.asarray(rows["action_loss_gradients"]["bc"], np.float64)[:, 0]
    bca = np.asarray(rows["action_loss_gradients"]["prospective_bca3"], np.float64)[:, 0]
    scored = high & (a[:, 0] < -KNEE)
    native = q + bc
    outward = scored & (native > 0)
    reversed_ = outward & (native + bca < 0)
    inward_bca = scored & (bca < 0)
    dose_rows = outward & (bca < 0)
    dose = KAPPA * native[dose_rows] / (-bca[dose_rows])
    med = lambda x: float(np.median(x)) if x.size else None          # noqa: E731
    return {"rows": int(len(a)), "high_load": int(high.sum()), "scored": int(scored.sum()),
            "native_outward": int(outward.sum()),
            "reversal_share": float(reversed_.sum() / outward.sum()) if outward.any() else None,
            "slope_ratio_median": med(np.abs(bca[scored]) / np.maximum(np.abs(q[scored]), 1e-12)),
            "inward_share": float(inward_bca.sum() / scored.sum()) if scored.any() else None,
            "dose_needed_median_kappa": med(dose), "dose_rows": int(dose_rows.sum()),
            "share_reversed_at_kappa_max": float(np.mean(dose <= KAPPA_MAX) * dose_rows.sum() / outward.sum()) if outward.any() and dose.size else 0.0,
            "signed_inward_over_q_median": med(-bca[scored] / np.maximum(np.abs(q[scored]), 1e-12)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    out = {"rule": "RULE.md", "rule_sha256": hashlib.sha256((HERE / "RULE.md").read_bytes()).hexdigest(), "runs": {}}
    for seed in (0, 1, 2):
        for arm in ("cutoff", "random"):
            for world in ("stall3", "nostall"):
                name = f"s{seed}_{arm}_{world}"
                for ckpt, kind in CHECKPOINTS.items():
                    path = RUN / f"{kind}_{name}" / f"gradient_audit_{ckpt}.json"
                    out["runs"].setdefault(name, {})[ckpt] = readout(path)
    stall = [f"s{s}_{a}_stall3" for s in (0, 1, 2) for a in ("cutoff", "random")]
    r20 = {n: out["runs"][n]["020000"] for n in stall}
    no_lev = all(r20[n]["reversal_share"] is not None and r20[n]["reversal_share"] < 0.25
                 and r20[n]["slope_ratio_median"] < 0.25 for n in stall)
    lev = any(all(r20[f"s{s}_{a}_stall3"]["reversal_share"] is not None and r20[f"s{s}_{a}_stall3"]["reversal_share"] >= 0.5
                  and r20[f"s{s}_{a}_stall3"]["inward_share"] >= 0.75 for s in (0, 1, 2)) for a in ("cutoff", "random"))
    out["verdict_20k"] = "NO LEVERAGE" if no_lev else ("LEVERAGE" if lev else "PARTIAL")
    out["placement_stall_minus_nostall_20k"] = {
        f"s{s}_{a}": {k: (out["runs"][f"s{s}_{a}_stall3"]["020000"][k] - out["runs"][f"s{s}_{a}_nostall"]["020000"][k])
                      for k in ("inward_share", "signed_inward_over_q_median")}
        for s in (0, 1, 2) for a in ("cutoff", "random")}
    (HERE / "leverage_result.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("verdict at 20k:", out["verdict_20k"])
    keys = ("scored", "native_outward", "reversal_share", "slope_ratio_median", "inward_share", "dose_needed_median_kappa",
            "share_reversed_at_kappa_max")
    for n in sorted(out["runs"]):
        for ckpt in CHECKPOINTS:
            r = out["runs"][n][ckpt]
            print(f"{n:22s} {ckpt}", "  ".join(f"{k}={r[k]:.3f}" if isinstance(r[k], float) else f"{k}={r[k]}" for k in keys))
    print("placement (stall - nostall, 20k):", json.dumps(out["placement_stall_minus_nostall_20k"], indent=None))


if __name__ == "__main__":
    main()

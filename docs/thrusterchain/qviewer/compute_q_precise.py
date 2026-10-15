"""Re-run of compute_q.py with 7 significant digits instead of 4, written to runs/qviewer/precise/.

Why: at 4 significant digits a value such as -104.2 has a resolution of 0.1, which is more than 2% of the spread of
the 41 swept values at many states (up to 86 of 100 states in an episode), so the curves of the viewer would show
steps that are rounding, not the critic or the physics. The computation is compute_q.main() unchanged (same
loaders, same hash checks, same tail-sum and sweep checks); only the rounding function and the output folder differ.
After the run this script checks that every integer field (a_hat, a_best_true) and every 4-digit summary number
equals compute_q.py's own q_data.json, so the two files describe the same computation.

Writes /root/bca-shift/runs/qviewer/precise/q_data.json and q_checks.json (and copies them to the Windows qviewer
folder's precise/ subfolder). Writes nothing else.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/c/Users/David/dev/bca-shift-work/qviewer")
import compute_q as CQ  # noqa: E402


def sig7(x):
    a = np.asarray(x, np.float64)
    if a.ndim == 0:
        v = float(a)
        if not math.isfinite(v):
            raise ValueError("non-finite number in the output")
        return float(f"{v:.7g}") + 0.0
    return [sig7(v) for v in a]


def main():
    CQ.sig4 = sig7
    CQ.OUT = Path("/root/bca-shift/runs/qviewer/precise")
    CQ.WIN = Path("/mnt/c/Users/David/dev/bca-shift-work/qviewer/precise")
    CQ.OUT.mkdir(parents=True, exist_ok=True)
    CQ.WIN.mkdir(parents=True, exist_ok=True)
    status = CQ.main()
    p = CQ.OUT / "q_data.json"
    d = json.loads(p.read_text())
    d["schema"] = "bca-shift-qviewer-v1-precise"
    d["note"] = [n.replace("4 significant digits", "7 significant digits") for n in d["note"]]
    d["note"].append("Same computation as compute_q.py (run by compute_q_precise.py), rounded to 7 significant digits.")
    # same computation as the 4-digit file?
    ref = json.loads(Path("/root/bca-shift/runs/qviewer/q_data.json").read_text())
    F = {f: i for i, f in enumerate(ref["episode_fields"])}
    same = dict(episodes=len(d["episodes"]) == len(ref["episodes"]), indices=True)
    s4 = lambda v: float(f"{v:.4g}") + 0.0
    worst, mism = 0.0, 0
    for a, b in zip(d["episodes"], ref["episodes"]):
        same["indices"] &= a[F["a_hat"]] == b[F["a_hat"]] and a[F["a_best_true"]] == b[F["a_best_true"]]
        for key in ("return", "discounted_return", "points", "state", "cmd", "reward", "push", "q_actor", "qpi_actor",
                    "q_sweep", "qpi_sweep"):
            x = np.asarray(a[F[key]], np.float64).ravel()
            y = np.asarray(b[F[key]], np.float64).ravel()
            mism += int(np.sum(np.array([s4(v) for v in x]) != y))
            worst = max(worst, float(np.max(np.abs(x - y) / np.maximum(np.abs(y), 1e-300))))
    sworst, smism = 0.0, 0
    for a, b in zip(d["summaries"], ref["summaries"]):
        same["indices"] &= a[3]["critic_rises_while_truth_falls_steps"] == b[3]["critic_rises_while_truth_falls_steps"]
        for k, v in a[3].items():
            if isinstance(v, float):
                w = b[3][k]
                smism += int(s4(v) != w)
                sworst = max(sworst, abs(v - w) / max(abs(w), 1e-300))
            elif not isinstance(v, list):
                same["indices"] &= v == b[3][k]
    # 4-digit rounding moves a value by at most half a unit in its 4th digit: relative 5e-4
    same["values_within_4_digit_rounding"] = bool(worst <= 5e-4 and sworst <= 5e-4)
    d["same_as_4_digit_file"] = dict(same, worst_relative_difference=worst, values_rounding_differently=mism,
                                     summary_worst_relative_difference=sworst, summary_rounding_differently=smism)
    text = json.dumps(d, separators=(",", ":"), allow_nan=False)
    p.write_text(text)
    (CQ.WIN / "q_data.json").write_text(text)
    print(json.dumps(d["same_as_4_digit_file"]))
    return status if all(same.values()) else 2


if __name__ == "__main__":
    sys.exit(main())

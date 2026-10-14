"""Score the return test against its frozen pass marks (channel/PASSMARKS_training_test.md). Read only.

Returns: the new arms' exact 100-step returns (command_le_train.py evaluate), and the references' from the version-5
records: the no-caution actor and the exact oracle lower edge (runs/shift_iw_v5/<quality>/evaluate/seed<k>/
room_lower_edge.json), version 5's unweighted and exact-strong lower edges (runs/shift_iw_v5_explore/<quality>/
evaluate/seed<k>/bca_lower_edge.json). Writes runs/shift_iw_v5_explore/channel/train/SCORE.json and prints a table.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

V5 = Path("/root/bca-shift/runs/shift_iw_v5")
EX = Path("/root/bca-shift/runs/shift_iw_v5_explore")
TR = EX / "channel" / "train"
QUALITIES = ("medium-expert", "expert")
SEEDS = (0, 1, 2, 3, 4)
LAWS = ("strong", "mild", "none")
BAR = 1.8
T = float(stats.t.ppf(0.95, 4))


def returns(quality, seed):
    out = {}
    ev = json.loads((TR / quality / f"evaluate_seed{seed}.json").read_text())["evaluations"]
    out.update({k: v for k, v in ev.items()})
    room = json.loads((V5 / quality / "evaluate" / f"seed{seed}" / "room_lower_edge.json").read_text())["evaluations"]
    out["no_caution"], out["oracle"] = room["le_none"], room["le_oracle"]
    le = json.loads((EX / quality / "evaluate" / f"seed{seed}" / "bca_lower_edge.json").read_text())["evaluations"]
    out["v5_unweighted"], out["v5_exact_strong"] = le["le_bca_unweighted"], le["le_bca_exact_strong"]
    return out


def paired(a, b, quality, law, field="J"):
    d = np.array([returns(quality, s)[a][law][field] - returns(quality, s)[b][law][field] for s in SEEDS])
    m, sd = float(d.mean()), float(d.std(ddof=1))
    return dict(mean=m, lower=m - T * sd / math.sqrt(len(d)), per_seed=d.tolist())


def main():
    score, lines = {}, []
    for q in QUALITIES:
        s = {}
        for a, b in (("cmd_strong", "v5_unweighted"), ("cmd_strong", "untargeted"), ("cmd_strong", "cmd_train"),
                     ("cmd_strong", "no_caution"), ("oracle", "no_caution"), ("cmd_train", "v5_unweighted"),
                     ("untargeted", "v5_unweighted"), ("v5_exact_strong", "v5_unweighted")):
            for law in LAWS:
                s[f"{a} - {b} | {law}"] = dict(J=paired(a, b, q, law), points=paired(a, b, q, law, "points"))
        t1 = s["cmd_strong - v5_unweighted | strong"]["J"]
        t2 = s["cmd_strong - untargeted | strong"]["J"]
        t3 = s["cmd_strong - v5_unweighted | none"]["J"]
        gain, room = s["cmd_strong - no_caution | strong"]["J"]["mean"], s["oracle - no_caution | strong"]["J"]["mean"]
        s["marks"] = dict(T1=bool(t1["mean"] >= BAR and t1["lower"] > 0), T2=bool(t2["mean"] >= BAR and t2["lower"] > 0),
                          T3=bool(t3["mean"] >= -BAR), share_of_oracle_gain=gain / room if room else None)
        s["mean_points"] = {arm: {law: float(np.mean([returns(q, sd)[arm][law]["points"] for sd in SEEDS]))
                                  for law in LAWS}
                            for arm in ("no_caution", "v5_unweighted", "v5_exact_strong", "cmd_strong", "cmd_train",
                                        "untargeted", "oracle")}
        score[q] = s
        lines.append(f"== {q}")
        for k, v in s.items():
            if k in ("marks", "mean_points"):
                continue
            lines.append(f"  {k:44s} J {v['J']['mean']:+8.1f} (lower {v['J']['lower']:+8.1f})   points "
                         f"{v['points']['mean']:+6.2f}   per seed " + ", ".join(f"{x:+.1f}" for x in v['J']['per_seed']))
        lines.append("  mean points (strong / mild / none):")
        for arm, v in s["mean_points"].items():
            lines.append(f"    {arm:16s} {v['strong']:6.2f} {v['mild']:6.2f} {v['none']:6.2f}")
        lines.append(f"  MARKS: T1 {'met' if s['marks']['T1'] else 'MISSED'}, T2 {'met' if s['marks']['T2'] else 'MISSED'}, "
                     f"T3 {'met' if s['marks']['T3'] else 'MISSED'}; share of the oracle's gain (strong) "
                     f"{s['marks']['share_of_oracle_gain']:.2f}")
    (TR / "SCORE.json").write_text(json.dumps(score, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

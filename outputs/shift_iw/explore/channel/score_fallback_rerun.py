"""Score the rerun against its frozen pass marks (channel/PASSMARKS_fallback_rerun.md). Read only.

Adds the rerun's arms (runs/shift_iw_v5_explore/channel/train_f1/<quality>/evaluate_seed<k>.json) to the first
test's returns (score_training_test.returns) and writes train_f1/SCORE.json.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_training_test as S  # noqa: E402

F1 = S.EX / "channel" / "train_f1"


def returns(quality, seed):
    out = S.returns(quality, seed)
    out.update(json.loads((F1 / quality / f"evaluate_seed{seed}.json").read_text())["evaluations"])
    return out


def paired(a, b, quality, law, field="J"):
    d = np.array([returns(quality, s)[a][law][field] - returns(quality, s)[b][law][field] for s in S.SEEDS])
    m, sd = float(d.mean()), float(d.std(ddof=1))
    return dict(mean=m, lower=m - S.T * sd / math.sqrt(len(d)), per_seed=d.tolist())


def main():
    score, lines = {}, []
    for q in S.QUALITIES:
        s = {}
        for a, b in (("A2_cmd_strong", "v5_unweighted"), ("A2_cmd_strong", "C2_untargeted"),
                     ("A2_cmd_strong", "cmd_strong"), ("A2_cmd_strong", "B2_cmd_train"),
                     ("A2_cmd_strong", "no_caution"), ("C2_untargeted", "v5_unweighted"),
                     ("oracle", "no_caution")):
            for law in S.LAWS:
                s[f"{a} - {b} | {law}"] = dict(J=paired(a, b, q, law), points=paired(a, b, q, law, "points"))
        u1 = s["A2_cmd_strong - v5_unweighted | strong"]["J"]
        u2 = s["A2_cmd_strong - C2_untargeted | strong"]["J"]
        u3 = s["A2_cmd_strong - v5_unweighted | none"]["J"]
        if q == "medium-expert":
            u4 = s["A2_cmd_strong - cmd_strong | strong"]["J"]["mean"] >= 50.0
        else:
            u4 = u1["mean"] >= 39.3
        gain = s["A2_cmd_strong - no_caution | strong"]["J"]["mean"]
        room = s["oracle - no_caution | strong"]["J"]["mean"]
        s["marks"] = dict(U1=bool(u1["mean"] >= S.BAR and u1["lower"] > 0),
                          U2=bool(u2["mean"] >= S.BAR and u2["lower"] > 0), U3=bool(u3["mean"] >= -S.BAR),
                          U4=bool(u4), share_of_oracle_gain=gain / room)
        s["mean_points"] = {arm: {law: float(np.mean([returns(q, sd)[arm][law]["points"] for sd in S.SEEDS]))
                                  for law in S.LAWS}
                            for arm in ("no_caution", "v5_unweighted", "cmd_strong", "A2_cmd_strong", "B2_cmd_train",
                                        "C2_untargeted", "oracle")}
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
        m = s["marks"]
        lines.append("  MARKS: " + ", ".join(f"{k} {'met' if m[k] else 'MISSED'}" for k in ("U1", "U2", "U3", "U4")) +
                     f"; share of the oracle's gain (strong) {m['share_of_oracle_gain']:.2f}")
    (F1 / "SCORE.json").write_text(json.dumps(score, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

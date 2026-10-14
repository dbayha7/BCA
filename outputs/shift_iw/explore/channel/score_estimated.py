"""Score the estimated-weights test against its frozen pass marks (channel/PASSMARKS_estimated_weights.md). Read only.

V1, V2 from train_est/<quality>/checks_seed<k>.json; X1-X4 from the arms' returns (train_est/<quality>/
evaluate_seed<k>.json) against the references of score_fallback_rerun.returns. Writes train_est/SCORE.json.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_fallback_rerun as SF  # noqa: E402

S = SF.S
EST = S.EX / "channel" / "train_est"


def returns(quality, seed):
    out = SF.returns(quality, seed)
    out.update(json.loads((EST / quality / f"evaluate_seed{seed}.json").read_text())["evaluations"])
    return out


def paired(a, b, quality, law, field="J"):
    d = np.array([returns(quality, s)[a][law][field] - returns(quality, s)[b][law][field] for s in S.SEEDS])
    m, sd = float(d.mean()), float(d.std(ddof=1))
    return dict(mean=m, lower=m - S.T * sd / math.sqrt(len(d)), per_seed=d.tolist())


def main():
    score, lines = {}, []
    for q in S.QUALITIES:
        ck = [json.loads((EST / q / f"checks_seed{s}.json").read_text()) for s in S.SEEDS]
        v1 = float(np.mean([c["V1"]["all"]["covered_or_refused"] for c in ck]))
        agree = float(np.mean([c["V2"]["agreement"] for c in ck]))
        past = [c["V2"]["exact_no_credit_past_80_also_estimated"] for c in ck]
        past_m = float(np.mean([p for p in past if p is not None]))
        s = dict(V1_covered_or_refused=v1, V2_agreement=agree, V2_past80_also_estimated=past_m,
                 V2_past80_per_seed=past)
        for a, b in (("A3_estimated", "v5_unweighted"), ("A3_estimated", "C3_untargeted"),
                     ("A3_estimated", "A2_cmd_strong"), ("A2_cmd_strong", "v5_unweighted"),
                     ("C3_untargeted", "v5_unweighted"), ("A3_estimated", "no_caution"), ("oracle", "no_caution")):
            for law in S.LAWS:
                s[f"{a} - {b} | {law}"] = dict(J=paired(a, b, q, law), points=paired(a, b, q, law, "points"))
        x1 = s["A3_estimated - v5_unweighted | strong"]["J"]
        x2 = s["A3_estimated - C3_untargeted | strong"]["J"]
        x3 = s["A3_estimated - v5_unweighted | none"]["J"]
        a2 = s["A2_cmd_strong - v5_unweighted | strong"]["J"]["mean"]
        s["marks"] = dict(V1=v1 >= 0.88, V2=bool(agree >= 0.90 and past_m >= 0.80),
                          X1=bool(x1["mean"] >= S.BAR and x1["lower"] > 0),
                          X2=bool(x2["mean"] >= S.BAR and x2["lower"] > 0), X3=bool(x3["mean"] >= -S.BAR),
                          X4=bool(x1["mean"] >= 0.5 * a2), share_of_A2_gain=x1["mean"] / a2,
                          share_of_oracle_gain=s["A3_estimated - no_caution | strong"]["J"]["mean"] /
                          s["oracle - no_caution | strong"]["J"]["mean"])
        s["mean_points"] = {arm: {law: float(np.mean([returns(q, sd)[arm][law]["points"] for sd in S.SEEDS]))
                                  for law in S.LAWS}
                            for arm in ("no_caution", "v5_unweighted", "A2_cmd_strong", "A3_estimated",
                                        "C3_untargeted", "oracle")}
        score[q] = s
        lines.append(f"== {q}: V1 {v1:.3f}, V2 agreement {agree:.3f}, past-80% also estimated {past_m:.3f}")
        for k, v in s.items():
            if not isinstance(v, dict) or "J" not in v:
                continue
            lines.append(f"  {k:44s} J {v['J']['mean']:+8.1f} (lower {v['J']['lower']:+8.1f})   points "
                         f"{v['points']['mean']:+6.2f}   per seed " + ", ".join(f"{x:+.1f}" for x in v['J']['per_seed']))
        lines.append("  mean points (strong / mild / none):")
        for arm, v in s["mean_points"].items():
            lines.append(f"    {arm:16s} {v['strong']:6.2f} {v['mild']:6.2f} {v['none']:6.2f}")
        m = s["marks"]
        lines.append("  MARKS: " + ", ".join(f"{k} {'met' if m[k] else 'MISSED'}" for k in ("V1", "V2", "X1", "X2", "X3",
                                                                                               "X4")) +
                     f"; share of A2's gain {m['share_of_A2_gain']:.2f}; of the oracle's {m['share_of_oracle_gain']:.2f}")
    (EST / "SCORE.json").write_text(json.dumps(score, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

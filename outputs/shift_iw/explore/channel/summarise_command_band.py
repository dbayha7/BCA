"""Table of command_band_check.py's results (means over seeds, min to max) and the pass marks C1-C4. Read only."""
import json
from pathlib import Path

import numpy as np

OUT = Path("/root/bca-shift/runs/shift_iw_v5_explore/channel/command_band")
QUALITIES = ("random", "medium-replay", "medium", "medium-expert", "expert")


def get(r, *path):
    for p in path:
        r = r.get(p) if isinstance(r, dict) else None
    return r


def stats(vals):
    v = np.array([x for x in vals if x is not None], float)
    return (v.mean(), v.min(), v.max()) if len(v) else None


def fmt(vals, pct=False, nd=1):
    s = stats(vals)
    if s is None:
        return "n/a"
    k = 100.0 if pct else 1.0
    return f"{k * s[0]:.{nd}f} ({k * s[1]:.{nd}f} to {k * s[2]:.{nd}f})"


marks = {}
for q in QUALITIES:
    S = json.loads((OUT / f"{q}.json").read_text())["seeds"]
    col = lambda *p: [get(s, *p) for s in S]
    print("==", q, f"({len(S)} seeds)")
    lines = {
        "bank weights: Kish share": fmt(col("bank_weights", "kish_share"), True, 2),
        "threshold with no test atom / R_u": fmt(col("thresholds", "no_atom_over_R_u"), nd=2),
        "largest finite threshold (cap) / R_u": fmt(col("thresholds", "cap_over_R_u"), nd=2),
        "C1 test draw: covered or refused": fmt(col("at_test_draw", "all", "covered_or_refused"), True),
        "   test draw: refused": fmt(col("at_test_draw", "all", "refused"), True),
        "C2 at actor within 80%: covered or refused": fmt(col("at_actor", "within_80", "covered_or_refused"), True),
        "C2 at actor past 80%: covered or refused": fmt(col("at_actor", "past_80", "covered_or_refused"), True),
        "   at actor past 80%: refused": fmt(col("at_actor", "past_80", "refused"), True),
        "   at actor within 80%: covered among certified": fmt(col("at_actor", "within_80", "covered_among_certified"), True),
        "C3 at actor within 80%: refused": fmt(col("at_actor", "within_80", "refused"), True),
        "     because inside the removed ball": fmt(col("refusal_reasons_at_actor", "within_80", "in_ball"), True),
        "     because the recorder density is 0": fmt(col("refusal_reasons_at_actor", "within_80",
                                                         "recorder_density_zero"), True),
        "       (command at exactly 80%, not stalled)": fmt(col("refusal_reasons_at_actor", "within_80",
                                                               "at_or_past_knee_not_stalled"), True),
        "     because the test mass is past the onset": fmt(col("refusal_reasons_at_actor", "within_80",
                                                               "mass_beyond_onset"), True),
        "C3 within-80% width / version 5 exact-strong": fmt(col("within_80_width_over_v5", "mean"), nd=2),
        "C4 check (e) passes (seeds)": f"{sum(bool(get(s, 'check_e', 'passed')) for s in S)} of {len(S)}",
        "C4 best throttle past 80%: Q1 - W": fmt(col("check_e", "best_past80_share"), True),
        "C4 best throttle past 80%: Q1 alone": fmt(col("check_e", "q1_alone_best_past80_share"), True),
        "   actor past 80% share of rows": fmt(col("actor_past_80_share"), True),
    }
    for k, v in lines.items():
        print(f"  {k:52s} {v}")
    m = lambda *p: stats(col(*p))[0] if stats(col(*p)) else None
    mk = dict(C1=m("at_test_draw", "all", "covered_or_refused") >= 0.88)
    if q in ("medium-expert", "expert"):
        mk["C2"] = (m("at_actor", "within_80", "covered_or_refused") >= 0.88 and
                    m("at_actor", "past_80", "covered_or_refused") >= 0.88)
        mk["C3"] = (m("at_actor", "within_80", "refused") <= 0.10 and
                    (m("within_80_width_over_v5", "mean") or 0) <= 1.5)
        passes = sum(bool(get(s, "check_e", "passed")) for s in S)
        mk["C4"] = (passes >= 4 and m("check_e", "best_past80_share") <= 0.5 * m("check_e", "q1_alone_best_past80_share"))
    marks[q] = mk
print("== pass marks")
for q, mk in marks.items():
    print(f"  {q:14s} " + ", ".join(f"{k} {'met' if v else 'MISSED'}" for k, v in mk.items()))

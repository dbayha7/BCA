"""Summarise the outside-gate shift cards over seeds: one table per situation (reported only; no pass rule uses it).

Usage (from the repository root): python outputs/shift/stop-report/cards/summarise_cards.py outputs/shift/stop-report/cards
Writes cards_summary.json and cards_summary.md next to the cards.
"""
import json
import sys
from pathlib import Path

import numpy as np

SITUATIONS = [("recorded", "Recorded conditions"), ("strong_current", "Strong current"), ("big_waves", "Big waves"),
              ("big_kicks", "Big kicks")]
POLICIES = ["recording controller", "stall-aware controller", "rising-thrust controller", "cloned actor",
            "plain TD3+BC on the gapped log", "actor of the full-log critic (H2)", "actor of the gapped critic (H2)",
            "room check, no extra caution", "room check, harm-placed"]
MEASURES = [("points", lambda c: c["points"]["policy"], "{:.1f}"),
            ("past 80%", lambda c: c["unrecorded_throttle_share"], "{:.2f}"),
            ("outside recorded", lambda c: c["outside_recorded_conditions"]["share"], "{:.2f}"),
            ("removed share", lambda c: c["removed_share"]["mean"], "{:.2f}"),
            ("loss from past-80% steps", lambda c: c["points"]["loss_from_unrecorded_throttle"], "{:.1f}"),
            ("distance to data", lambda c: c["distance_to_data"]["visited"]["p50"], "{:.2f}")]


def main(folder):
    folder = Path(folder)
    cards = [json.loads(p.read_text()) for p in sorted(folder.glob("cards_seed*.json"))]
    seeds = [c["rep"] for c in cards]
    summary = dict(kind="shift-cards-summary", seeds=seeds, prereg_sha256=cards[0]["prereg_sha256"],
                   note="means over seeds of the outside-gate shift cards; reported only", situations={})
    lines = ["Means over seeds " + ", ".join(map(str, seeds)) + ". Columns: points; share of steps past 80% on any "
             "thruster; share of steps outside the recorded conditions; mean removed share m(s) at the visited states "
             "(how often the recording controller's command there fell in the removed region); points lost at the "
             "past-80% steps; median distance from the visited (state, command) pairs to the 10th nearest training "
             "row, in standard units (held-out recorded rows: " + "{:.2f}".format(
                 np.mean([c["cards"]["recording controller"]["recorded"]["distance_to_data"]["heldout_log_rows"]["p50"]
                          for c in cards])) + ").", ""]
    for key, label in SITUATIONS:
        rows = {}
        for pol in POLICIES:
            rows[pol] = {name: float(np.mean([f(c["cards"][pol][key]) for c in cards])) for name, f, _ in MEASURES}
        summary["situations"][key] = rows
        lines += [f"**{label}**", "", "| Controller | " + " | ".join(n for n, _, _ in MEASURES) + " |",
                  "|---|" + "---|" * len(MEASURES)]
        for pol, v in rows.items():
            lines.append(f"| {pol} | " + " | ".join(fmt.format(v[n]) for n, _, fmt in MEASURES) + " |")
        lines.append("")
    (folder / "cards_summary.json").write_text(json.dumps(summary, indent=1))
    (folder / "cards_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])

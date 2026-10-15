"""Record the channel check (2026-10-14, exploratory, no training) in /root/bca-shift/EXPERIMENTS.md. Exact, asserted."""
from pathlib import Path

P = Path("/root/bca-shift/EXPERIMENTS.md")
t = P.read_text(encoding="utf-8")

reps = [
    ("Last updated: 2026-10-13.", "Last updated: 2026-10-14."),
    ("| Weighting fixes coverage under the known shift (83–87% → 91–95%); it did not raise return |",
     "| Weighting fixes coverage under the known shift (83–87% → 91–95%); it did not raise return. The lower edge is "
     "the channel with room, but BCA's band is read where it was never calibrated (25–48% coverage past 80% at the top "
     "two qualities) |"),
    ("  - Next, if the user wants return gains: a band whose width grows at unsupported commands (for example read relative\n"
     "    to the recorded command, as Stage 1's oracle was), then a new pre-registered test.\n",
     "  - Next, if the user wants return gains: a band whose width grows at unsupported commands (for example read relative\n"
     "    to the recorded command, as Stage 1's oracle was), then a new pre-registered test.\n"
     "- **The user's request (2026-10-14):** \"i wanr return gain as well. its good that the coverage aspect is working\n"
     "  accordingly, but we need to identify the right channel for the penalty to actually impact policy performance\".\n"
     "- **Channel check (2026-10-14, exploratory, post hoc, no training;** expectations written first:\n"
     "  `outputs/shift_iw/explore/channel/`).\n"
     "  - **Which channel:** the room checks with exact information already answer it.\n"
     "    - The lower edge (a penalty on the value at the actor's own command) gains +5.8 points (medium-expert) and\n"
     "      +8.4 (expert) under the strong shift.\n"
     "    - The capped pull gains at most 0.2 points even when placed perfectly.\n"
     "    - On the three lower qualities neither channel has room: the perfect lower edge loses 0.8–1.1 points.\n"
     "  - **Why BCA's lower edge did not move return:** the band is honest where it was calibrated, not where the lower\n"
     "    edge reads it.\n"
     "    - At the no-caution actor's own commands within 80%, it covers 95–96% at the top two qualities.\n"
     "    - The actor pushes past 80% on 53–62% of hard-water rows. There the critic overrates its command by 3.4\n"
     "      (medium-expert) and 1.7 (expert) per step, and the band covers 48% and 25% (strong law). It would have to be\n"
     "      3.4× and 8.1× wider, but it widens only 15–25% past 80%.\n"
     "    - On the three lower qualities the band is 10–18× wider than on expert data, and the actors rarely go past\n"
     "      80%, so it covers 81–93% at their commands.\n"
     "    - Expectations: E1 met; E2 and E3 partly met (they hold at the top two qualities only, and the weighting helped\n"
     "      coverage more than predicted: +6 to +10 points).\n"
     "  - **Why weighting by the current law cannot raise return:** it multiplies every row's band by a nearly constant\n"
     "    factor (1.45× expert, 1.75× medium-expert). In the room checks, multiplying perfectly informed caution by 1.5\n"
     "    moves return by 0.0–0.7 points. Weighting has to change where the band is wide, not only how wide it is.\n"
     "  - **Consequence:** for weighting to reach return, the weights must describe where the band is read, which is the\n"
     "    actor's commands under the test law. Both parts are known exactly in this world: the current law, and the\n"
     "    recording controller's command density in closed form. Proposed to the user (decision 10).\n"),
    ("10. Known-shift weighting (entry 17): whether to build a band that widens at unsupported commands and pre-register a\n"
     "   new return test.\n",
     "10. Known-shift weighting (entry 17): the user wants a return gain (2026-10-14). The channel check points to the\n"
     "   lower edge, with weights for the actor's commands under the test law. Waiting for the user's go:\n"
     "   - first, a no-training check of that band, with pass marks written first;\n"
     "   - then an actor-only training test on medium-expert and expert.\n"),
]
for old, new in reps:
    assert t.count(old) == 1, (t.count(old), old[:80])
    t = t.replace(old, new)

log_tail = ("    exploratory BCA results. Page and sources: `docs/thrusterchain/`.\n")
assert t.count(log_tail) == 1 and t.endswith(log_tail), "change log tail moved"
t += ("- **2026-10-14:** the user asked for a return gain from weighting and for the right channel. An exploratory check\n"
      "  (no training) found that the lower edge is the channel with room, but BCA's band covers only 25–48% at the\n"
      "  actor's past-80% commands at the top two qualities. The current-law weighting rescales the band nearly\n"
      "  uniformly (1.45–1.75×), and a rescale of that size moves return by at most 0.7 points even with perfect\n"
      "  information. Next: weights for the actor's commands under the test law (decision 10, waiting for the user).\n")
P.write_text(t, encoding="utf-8")
print("updated", P)

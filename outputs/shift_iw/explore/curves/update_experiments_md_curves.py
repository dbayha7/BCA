"""Record the learning curves (2026-10-14) in /root/bca-shift/EXPERIMENTS.md. Exact, asserted."""
from pathlib import Path

P = Path("/root/bca-shift/EXPERIMENTS.md")
t = P.read_text(encoding="utf-8")
reps = [
    ("  - Write-up: `outputs/shift_iw/explore/channel/RESULTS_estimated_weights.md`.\n",
     "  - Write-up: `outputs/shift_iw/explore/channel/RESULTS_estimated_weights.md`.\n"
     "- **Learning curves for every setup (2026-10-14; the user's request \"im going to need to see learning curves as well\n"
     "  comprehensively for all possible env setups\" and go \"Full set, ~3 h CPU\").**\n"
     "  - **Method:** every run of this test was retrained with checkpoints (actor-only every 1,000 of 20,000 steps; full\n"
     "    TD3+BC every 5,000 of 100,000 updates), and every checkpoint was scored on the three start banks.\n"
     "  - **Reproduction:** all 330 reruns with an earlier run end at its saved actor bit for bit. Version 5's runs\n"
     "    reproduce only when pinned to one CPU, as they were trained.\n"
     "  - **New runs:** the command-weighted arms (A2, B2, C2, A3, C3) on the three lower qualities, with expectations\n"
     "    written first. Mostly met: within ±10 units except estimated weights on medium (−11.4); 3 of 6 controls beyond\n"
     "    ±5.\n"
     "  - **Shown:**\n"
     "    - On expert, actor-only training against the critic lowers return (the cloned actor's 95.2 points on the strong\n"
     "      test falls to 90.7). The command-weighted band prevents it and rises to 96.5.\n"
     "    - The first test's medium-expert collapse comes after a peak at 3,000 steps.\n"
     "    - Full TD3+BC on the lower qualities peaks at 5,000–15,000 updates and loses 4–17 points by 100,000; the\n"
     "      capped pull loses 9–23.\n"
     "  - Page: finding G (`docs/thrusterchain/`); code and data: `outputs/shift_iw/explore/curves/`.\n"),
    ("   With estimated weights the gain holds on expert (+2.3 points) and vanishes on medium-expert. In progress:\n"
     "   learning curves for all five qualities × three test conditions (the user's request), and the ThrusterChain\n"
     "   page.\n",
     "   With estimated weights the gain holds on expert (+2.3 points) and vanishes on medium-expert. Learning curves\n"
     "   for every setup are on the ThrusterChain page (finding G). Open: pushing; estimating support on mixed data.\n"),
]
for old, new in reps:
    assert t.count(old) == 1, (t.count(old), old[:70])
    t = t.replace(old, new)
t += ("  Learning curves for all 5 qualities × 3 tests: every run retrained with checkpoints (330 of 330 reproduce their\n"
      "  saved actor bit for bit) and the command-weighted arms run on the three lower qualities. The ThrusterChain page,\n"
      "  version 16, adds them as finding G.\n")
P.write_text(t, encoding="utf-8")
print("updated", P)

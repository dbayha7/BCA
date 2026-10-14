"""Record the fallback rerun (2026-10-14) in /root/bca-shift/EXPERIMENTS.md. Exact, asserted."""
from pathlib import Path

P = Path("/root/bca-shift/EXPERIMENTS.md")
t = P.read_text(encoding="utf-8")

reps = [
    ("| Weighting fixes coverage under the known shift (83–87% → 91–95%). Weighting the band for the actor's own "
     "commands, with refusals as \"no value credit\", raises return on expert data by 6.2 points under the strong "
     "shift (73% of the exact-information room); on medium-expert it loses 9.8 |",
     "| Weighting fixes coverage under the known shift (83–87% → 91–95%). Weighting the band for the actor's own "
     "commands, with \"no value credit\" where the data lack the command, raises return under the strong shift by "
     "2.0 points (medium-expert) and 5.8 (expert), every seed, with no cost without the shift |"),
    ("  - Write-up: `outputs/shift_iw/explore/channel/RESULTS_training_test.md`.\n",
     "  - Write-up: `outputs/shift_iw/explore/channel/RESULTS_training_test.md`.\n"
     "- **Rerun with the refusal fallback fixed (2026-10-14; the user's go: \"Fix the fallback, then rerun\n"
     "  (Recommended)\").** The pass marks were frozen first (`PASSMARKS_fallback_rerun.md`, SHA-256 `8b8f7a27…`).\n"
     "  - **The change:** \"no value credit\" applies only where the exact ratio is infinite (the data never contain the\n"
     "    command at that state and episode, or it is in the removed hold). A refusal because the bank is too small\n"
     "    gets version 5's capped threshold. Checks before training passed (bit-equal step; refusal types match).\n"
     "  - **U1–U4 met on both qualities.** Against version 5's unweighted lower edge under the strong shift:\n"
     "    - medium-expert +25.0 return units (lower bound 22.6; 2.0 points; every seed +22.8 to +28.8);\n"
     "    - expert +74.2 (lower bound 68.3; 5.8 points); that is 34% and 69% of the exact oracle's gain.\n"
     "    An untargeted control of the same amount gains about 1. Without the shift return also rises (+4.5 and\n"
     "    +15.6). Medium-expert's loss is gone (+149.8 over the first test).\n"
     "  - **The current-law part of the weight adds nothing to return** (A2 minus B2: −0.2 and +0.5). The gain comes\n"
     "    from weighting for the actor's commands, which here acts as a calibrated support check: it refuses where\n"
     "    the recorded commands never go and certifies elsewhere, with no knee or threshold given to it.\n"
     "  - Write-up: `outputs/shift_iw/explore/channel/RESULTS_fallback_rerun.md`.\n"),
    ("10. Known-shift weighting (entry 17): the return test met its marks on expert data and missed them on\n"
     "   medium-expert, where the refusal fallback (the recorded command) is random half the time. Waiting for the\n"
     "   user: fix the fallback on mixed data, or write up the result as it stands.\n",
     "10. Known-shift weighting (entry 17): the return gain is shown (rerun, U1–U4 met on medium-expert and expert).\n"
     "   Open: estimated instead of exact weights; whether to fold the result into the ThrusterChain page; commits.\n"),
]
for old, new in reps:
    assert t.count(old) == 1, (t.count(old), old[:80])
    t = t.replace(old, new)
t += ("  The user chose to fix the fallback: \"no value credit\" only where the data lack the command (infinite exact\n"
      "  ratio), version 5's capped threshold where the bank is merely too small. The rerun met U1–U4 on both\n"
      "  qualities: +25.0 return units (medium-expert) and +74.2 (expert) under the strong shift, with every seed\n"
      "  positive, no gain for an untargeted control of the same size, and no cost without the shift.\n")
P.write_text(t, encoding="utf-8")
print("updated", P)

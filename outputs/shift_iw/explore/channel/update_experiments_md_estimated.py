"""Record the estimated-weights test (2026-10-14) in /root/bca-shift/EXPERIMENTS.md. Exact, asserted."""
from pathlib import Path

P = Path("/root/bca-shift/EXPERIMENTS.md")
t = P.read_text(encoding="utf-8")

reps = [
    ("  - Write-up: `outputs/shift_iw/explore/channel/RESULTS_fallback_rerun.md`.\n",
     "  - Write-up: `outputs/shift_iw/explore/channel/RESULTS_fallback_rerun.md`.\n"
     "- **Estimated instead of exact weights (2026-10-14; the user's go: \"Proceed with all three\").** The pass marks\n"
     "  were frozen first (`PASSMARKS_estimated_weights.md`, SHA-256 `4560769a…`).\n"
     "  - **Weights:** the current-law part is version 5's logistic-regression estimate. The command part is a\n"
     "    classifier separating recorded commands from the actor's command plus its spread, refitted every 4,000 steps.\n"
     "  - **Refusal:** \"no value credit\" applies where the estimated odds exceed those of every recorded command, or the\n"
     "    command is in the removed ball. Nothing exact about the recording process or the laws is used.\n"
     "  - **Expert: V1, V2 and X1–X3 met; X4 missed.** +29.4 return units under the strong shift (lower bound 25.8,\n"
     "    2.3 points, every seed +25 to +35), against +0.9 for an untargeted control. That is 40% of the exact\n"
     "    weights' gain (the bar was half).\n"
     "  - **Medium-expert: V1 and X3 met; V2, X1, X2 and X4 missed.** +1.5 (lower bound +0.2). The estimated rule finds\n"
     "    only 5% of the past-80% commands the exact rule refuses, because those depend on the episode's random share,\n"
     "    which the classifier cannot see.\n"
     "  - **Meaning:** the estimated band stays valid (coverage held on both). The return gain depends on estimating\n"
     "    where the data lack a command, which is easy on clean expert data and hard on mixed data.\n"
     "  - Write-up: `outputs/shift_iw/explore/channel/RESULTS_estimated_weights.md`.\n"),
    ("   Open: estimated instead of exact weights; whether to fold the result into the ThrusterChain page; commits.\n",
     "   With estimated weights the gain holds on expert (+2.3 points) and vanishes on medium-expert. In progress:\n"
     "   learning curves for all five qualities × three test conditions (the user's request), and the ThrusterChain\n"
     "   page.\n"),
]
for old, new in reps:
    assert t.count(old) == 1, (t.count(old), old[:80])
    t = t.replace(old, new)
t += ("  With estimated weights (pass marks frozen first): expert keeps a gain of +29.4 return units (2.3 points) under\n"
      "  the strong shift, 40% of the exact weights' gain; medium-expert's gain vanishes (+1.5), because the classifier\n"
      "  cannot see which episodes contain random commands. The user then asked for learning curves for every setup;\n"
      "  every curve reruns the original training with checkpoints and checks it against the saved final actor.\n"
      "  Version 5's runs reproduce only when pinned to one CPU, as they were trained.\n")
P.write_text(t, encoding="utf-8")
print("updated", P)

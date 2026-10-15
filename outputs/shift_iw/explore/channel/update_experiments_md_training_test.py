"""Record the return test of the command-weighted lower edge (2026-10-14) in /root/bca-shift/EXPERIMENTS.md."""
from pathlib import Path

P = Path("/root/bca-shift/EXPERIMENTS.md")
t = P.read_text(encoding="utf-8")

reps = [
    ("| Weighting fixes coverage under the known shift (83–87% → 91–95%); it did not raise return. The lower edge is "
     "the channel with room, but BCA's band is read where it was never calibrated (25–48% coverage past 80% at the top "
     "two qualities) |",
     "| Weighting fixes coverage under the known shift (83–87% → 91–95%). Weighting the band for the actor's own "
     "commands, with refusals as \"no value credit\", raises return on expert data by 6.2 points under the strong "
     "shift (73% of the exact-information room); on medium-expert it loses 9.8 |"),
    ("  - **Next:** a return test of this lower edge on medium-expert and expert (arms: command-weighted under the strong\n"
     "    law; command-weighted under the training law; an untargeted control of the same amount). Its pass marks are\n"
     "    drafted and wait for the user's approval.\n",
     "- **Return test (2026-10-14; the user's approval: \"Approve as drafted (Recommended)\").** The pass marks were frozen\n"
     "  first (`outputs/shift_iw/explore/channel/PASSMARKS_training_test.md`, SHA-256 `204f53ff…`). Setup: actor-only\n"
     "  lower edge, version 5's protocol, medium-expert and expert, seeds 0–4, 30 runs on CPU (about 150 s each).\n"
     "  - **Checks before training:** all passed. With the weights switched off the step equals version 5's bit for bit,\n"
     "    and the in-graph density and refusals match the numpy check.\n"
     "  - **Expert: T1–T3 met.** A beats version 5's unweighted lower edge under the strong shift by 78.6 return units\n"
     "    (lower bound 72.8; 6.2 points; every seed +70 to +85), which is 73% of the exact oracle's gain. An untargeted\n"
     "    control with the same amount of \"no credit\" gains 1.1. Without the shift A gains 16.3.\n"
     "  - **Medium-expert: T1–T3 missed.** A loses 124.8 under the strong shift (9.8 points) and 35.6 without it.\n"
     "    - Post hoc: refusals pile up in the strongest third of currents (62–68%) and on random-command episodes.\n"
     "      59–65% of refused rows have a random recorded command, and \"no credit\" makes the actor copy them.\n"
     "  - **Knowing the test current law adds nothing to return:** A minus B (training law) is +1.0 on expert (lower\n"
     "    bound −0.3) and −15.1 on medium-expert. The gain comes from weighting for the actor's commands.\n"
     "  - Write-up: `outputs/shift_iw/explore/channel/RESULTS_training_test.md`.\n"),
    ("10. Known-shift weighting (entry 17): the user wants a return gain (2026-10-14). The command-weighted band is\n"
     "   honest at the actor's commands (C1–C2 met); C3 and C4 were missed. Waiting for the user's go on the return\n"
     "   test (`PASSMARKS_training_test_DRAFT.md`), which changes the lower edge's refusal rule to \"no credit over the\n"
     "   recorded command\".\n",
     "10. Known-shift weighting (entry 17): the return test met its marks on expert data and missed them on\n"
     "   medium-expert, where the refusal fallback (the recorded command) is random half the time. Waiting for the\n"
     "   user: fix the fallback on mixed data, or write up the result as it stands.\n"),
]
for old, new in reps:
    assert t.count(old) == 1, (t.count(old), old[:80])
    t = t.replace(old, new)
tail = "  \"no credit\" rule removes most of the past-80% preference. The return test's pass marks are drafted.\n"
assert t.endswith(tail)
t = t[:-len(tail)] + (
    "  \"no credit\" rule removes most of the past-80% preference. The return test (approved, pass marks frozen first)\n"
    "  met T1–T3 on expert data: +78.6 return units (6.2 points) under the strong shift, 73% of the exact oracle's\n"
    "  gain, and nothing for an untargeted control of the same size. It missed them on medium-expert (−124.8), where\n"
    "  refusals in strong currents make the actor copy random recorded commands. Knowing the test current law added\n"
    "  nothing to return; the gain comes from weighting for the actor's commands.\n")
P.write_text(t, encoding="utf-8")
print("updated", P)

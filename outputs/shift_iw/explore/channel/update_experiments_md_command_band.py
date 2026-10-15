"""Record the command-weighted band check (2026-10-14) in /root/bca-shift/EXPERIMENTS.md. Exact, asserted."""
from pathlib import Path

P = Path("/root/bca-shift/EXPERIMENTS.md")
t = P.read_text(encoding="utf-8")

reps = [
    ("    recording controller's command density in closed form. Proposed to the user (decision 10).\n",
     "    recording controller's command density in closed form. Proposed to the user (decision 10).\n"
     "- **Command-weighted band, check without training (2026-10-14; the user's go: \"Yes, build and check it\n"
     "  (Recommended)\").** Pass marks were written first (`outputs/shift_iw/explore/channel/`).\n"
     "  - **The band:** BCA's frozen band with one change, the weights. A (row, command) gets the exact ratio\n"
     "    w(c) × π_h(a | s) × Z(s) / β(a | s):\n"
     "    - π_h is the actor's command with BCA's own target-noise spread (0.2, clipped at 0.5);\n"
     "    - β is the recording controller's command density, exact for each episode's random share;\n"
     "    - Z is the share outside the removed ball.\n"
     "  - **C1, met on all five qualities:** at draws from that test distribution, 94–100% of commands are covered or\n"
     "    refused, by seed.\n"
     "  - **C2, met at the top two qualities:** at the actor's own commands, 98–100% are covered or refused. Every\n"
     "    past-80% command is refused, where version 5's band covered 25% and 48%.\n"
     "  - **C3, missed:** within 80% the band also refuses 25% (expert) and 48% (medium-expert) of the actor's commands.\n"
     "    15 and 9 points of that are commands inside the removed hold; the rest come from the small effective bank\n"
     "    (3–16%). On the three lower qualities it refuses 68–98%, a blanket refusal.\n"
     "  - **C4, missed:** version 5's refusal rule (the largest finite threshold, only 1.7–3.0× R_u) leaves the penalty\n"
     "    too small and flat from 80% to 81%. Check (e) passes on 0 and 1 of 5 seeds.\n"
     "  - **Post hoc, not a pass mark:** if a refused command gets no value credit over the recorded command, check (e)\n"
     "    passes on 5 of 5 seeds at both qualities. The share of hard-water rows whose best throttle is past 80% falls\n"
     "    from 80% to 1.8% (expert) and from 85% to 16.5% (medium-expert); the exact oracle reaches 34% and 52%.\n"
     "  - **Next:** a return test of this lower edge on medium-expert and expert (arms: command-weighted under the strong\n"
     "    law; command-weighted under the training law; an untargeted control of the same amount). Its pass marks are\n"
     "    drafted and wait for the user's approval.\n"),
    ("10. Known-shift weighting (entry 17): the user wants a return gain (2026-10-14). The channel check points to the\n"
     "   lower edge, with weights for the actor's commands under the test law. Waiting for the user's go:\n"
     "   - first, a no-training check of that band, with pass marks written first;\n"
     "   - then an actor-only training test on medium-expert and expert.\n",
     "10. Known-shift weighting (entry 17): the user wants a return gain (2026-10-14). The command-weighted band is\n"
     "   honest at the actor's commands (C1–C2 met); C3 and C4 were missed. Waiting for the user's go on the return\n"
     "   test (`PASSMARKS_training_test_DRAFT.md`), which changes the lower edge's refusal rule to \"no credit over the\n"
     "   recorded command\".\n"),
    ("  even for perfectly informed caution. Next: weights for the actor's commands under the test law (decision 10, "
     "waiting for the user).\n",
     "  even for perfectly informed caution. Next: weights for the actor's commands under the test law (decision 10).\n"
     "  With the user's go, that band was built and checked without training: it is honest where the lower edge reads\n"
     "  it and refuses every past-80% command, but version 5's refusal rule makes the penalty too weak. A post hoc\n"
     "  \"no credit\" rule removes most of the past-80% preference. The return test's pass marks are drafted.\n"),
]
for old, new in reps:
    assert t.count(old) == 1, (t.count(old), old[:80])
    t = t.replace(old, new)
P.write_text(t, encoding="utf-8")
print("updated", P)

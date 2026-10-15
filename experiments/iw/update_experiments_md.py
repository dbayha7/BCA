"""Add the 2026-10-13 results to /root/bca-shift/EXPERIMENTS.md: Stage 1's outcome (entry 16), the known-shift weighting
test (new entry 17), the status row, the decisions list and the change log. Exact, asserted replacements."""
from pathlib import Path

P = Path("/root/bca-shift/EXPERIMENTS.md")
t = P.read_text(encoding="utf-8")


def rep(old, new, count=1):
    global t
    n = t.count(old)
    assert n == count, f"expected {count}, found {n}: {old[:70]!r}"
    t = t.replace(old, new)


rep("Last updated: 2026-10-12.", "Last updated: 2026-10-13.")

rep("| 16 | ThrusterChain: a Gymnasium environment with a controlled distribution shift | Done: stopped before BCA (no room) "
    "| The shift costs 99 points in strong current, but caution of BCA's kind cannot reach it |",
    "| 16 | ThrusterChain: a Gymnasium environment with a controlled distribution shift | Done: stopped before BCA (no room); "
    "Stage 1 done (exploratory) | The shift costs 99 points in strong current; BCA's kind of caution cannot reach it, but "
    "a penalty on the value of the actor's own command can (93.9 of 98.9 points, exact oracle) |\n"
    "| 17 | ThrusterChain known-shift weighting test (spec version 5, five dataset qualities) | Done: claims 1–2 "
    "(coverage) hold; no quality qualified for BCA training; exploratory BCA arms run | Weighting fixes coverage under "
    "the known shift (83–87% → 91–95%); it did not raise return |")

rep("- **Known bug (reporting only):** `acceptance.report_cards`",
    "- **Stage 1 result (2026-10-13, exploratory, no BCA):** a penalty on the critic's value at the actor's own command,\n"
    "  built from the critic's exact one-step error relative to the recorded command, recovered 93.9 of the 98.9 points\n"
    "  lost in strong current and gained 7.9 to 19.6 points in the other three situations. Untargeted caution of the same\n"
    "  strength recovered at most 4.6. So caution reaches this harm when it acts on the value at the actor's own command,\n"
    "  given exact information. Details: the ThrusterChain page, section \"After the stop\".\n"
    "- **Known bug (reporting only):** `acceptance.report_cards`")

ENTRY17 = """### 17. ThrusterChain known-shift weighting test (spec version 5; claims 1–2 hold; BCA arms exploratory)
- **Why:** the user's goal, verbatim: "verify that bca's mechanics are working and then show importance weighting
  empirical showing an improvement given that we know the test distribution. This is all controlled, we need to know
  every value, so we can simulate a distribution shift."
- **Design** (`experiments/shift_iw`, spec version 5, SHA-256 `a71af9bd…`, frozen with its sidecar; the user approved
  it in chat, "Yes, freeze and run"):
  - The training current follows a tilted density on [0.1, 1.1] (λ = −2). The tests use λ = −2 (none), 0 (mild) and
    +2 (strong), so the exact weight ω(c) = p_test(c)/p_train(c) is known.
  - The expert's 80% hold is cut from the training and bank rows (the action gap), so its exact ratio is infinite and
    weighted BCA must refuse those steps.
  - Five D4RL-style dataset qualities: random, medium-replay, medium, medium-expert and expert.
  - Two uses of the band: the capped pull (BCA's dose) and the lower edge (−λ·mean[Q1 − k·[U(s, π(s)) − U(s, a_i)]_+]
    + BC, k = 1).
  - Per-step weights through a tabulated R(t).
  - Gates before any BCA arm: room checks with the exact error, and "can weighting move return" predictions.
- **Ran** 2026-10-13 in WSL: data, hosts, room checks, band, coverage, report and summary, every stage rc = 0. Run
  folder: `runs/shift_iw_v5/` (not committed).
- **Result, pre-registered:**
  - Mechanics hold on all five qualities: no-shift coverage 91.2–92.5%, width tracks the exact critic error within
    current bands, and the dose follows the width.
  - Under the strong shift, unweighted coverage falls to 83.3–86.7% and exact-weighted per-step coverage restores
    91.0–94.6%. Estimated weights land within 2 points of exact. Weighted BCA refuses 100% of steps using the removed
    hold and 0% of the others.
  - One side check failed: on medium-expert, the unweighted band was not over-confident inside the gap.
  - No quality qualified for a BCA training arm. The capped pull's room check fails on the low qualities, where
    placed caution lowers return by 2.9–4.6 points. At the top two qualities weighting moves the dose only 0.04–0.05
    (predicted gain 0.4 return units against 1.8 needed). The lower edge's perfect penalty helps at the top (+8.5
    expert, +5.8 medium-expert), but return is already level in its strength, and BCA's width at the actor's commands
    is 1/160 to 1/30 of the true error and barely correlated with it.
- **Exploratory, at the user's request ("Run the arms anyway, labelled exploratory"):**
  - **BCA arms** (`iw_explore/RESULTS.md` in the work folder; runs in `runs/shift_iw_v5_explore/`). On the strong
    starts, weighting changed return by −26.4 to +0.1 return units (capped pull) and −1.1 to +2.7 (lower edge). No arm
    gained 1.8 with a lower bound above 0. The predictions held at the top qualities.
    - The capped pull costs 5–9 points against plain TD3+BC on the three lower qualities, because it pulls toward
      mostly random recorded commands; weighting makes that slightly worse.
    - Points, strong starts: plain TD3+BC 66.0 / 77.0 / 84.1 / 91.1 / 90.3; capped pull, weighted 57.1 / 69.1 / 77.7
      / 91.5 / 90.3; lower edge, weighted 67.1 / 77.8 / 82.2 / 92.2 / 90.6 (random → expert).
  - **Plain TD3+BC coverage** with its two critics' disagreement as its band (the user's choice; expectations written
    first, `iw_explore/PLAIN_TWIN_PREREG.md`). It covers 38.6–48.1% at every shift. Its band is 0.22–0.32 times BCA's
    width and no wider at the removed hold. All four expectations were met; on medium-expert its coverage rose 1.4
    points under the shift instead of dropping or staying flat.
  - **Data fact** (confirmed from the training files): the version-5 data hold no thruster-1 commands past +80% in the
    expert data, and 3.6–9.9% in the other qualities, all random. The critic therefore guesses past 80% with or without
    the hold removed (the page's "critic in action" section shows it).
- **Meaning:**
  - BCA's mechanics are verified, and weighting fixes coverage under a known shift.
  - Weighting did not raise return here. The band is calibrated on recorded commands. Rescaling it cannot create the
    width that is missing at commands the data never contain, which is where return is lost.
  - Next, if the user wants return gains: a band whose width grows at unsupported commands (for example read relative
    to the recorded command, as Stage 1's oracle was), then a new pre-registered test.

"""
rep("## Decisions waiting on the user", ENTRY17 + "## Decisions waiting on the user")

rep("   test bed whose data carry a trace of it). Stage 2 (BCA) needs typed approval.",
    "   test bed whose data carry a trace of it). Stage 2 (BCA) needs typed approval.\n"
    "10. Known-shift weighting (entry 17): whether to build a band that widens at unsupported commands and pre-register a\n"
    "   new return test.")

rep("  - the ThrusterChain page gained a task overview and Half-Cheetah-style component sections.",
    "  - the ThrusterChain page gained a task overview and Half-Cheetah-style component sections;\n"
    "  - Stage 1 (exploratory): the exact-error penalty on the actor's own command recovered 93.9 of 98.9 points (entry 16);\n"
    "  - the known-shift weighting test (spec version 5, five qualities) was frozen with the user's approval and run.\n"
    "    Mechanics and weighted coverage hold; no quality qualified for BCA arms. The exploratory arms show no return gain\n"
    "    from weighting. Plain TD3+BC's twin-critic band covers 39–48% (entry 17);\n"
    "  - the ThrusterChain page went through versions 11–12: plain-language tooltips, specification sections first,\n"
    "    Results rewritten for newcomers, and the \"critic in action\" section added.")

P.write_text(t, encoding="utf-8")
print("updated", P)

"""Edit 4: the column descriptions (the tooltip script's COLS) of Results and of the weighting test, in table order.

Each table of a section takes the first unused entry whose column names include all of the table's own names, so the
entries follow the tables' order on the page. Entries kept from the page (the tables of "After the stop") are passed
through unchanged. Every description is a complete sentence. ta_texts.py records the same texts (EDIT4_RESULTS).
"""

REC = "in the recorded conditions (current 0.1 to 0.7, wave amplitude 0.1 to 0.4), the conditions of the training data"
STRONG = "in strong current (current 0.7 to 1.1, stronger than any in the data)"
WAVES = "in big waves (wave amplitude 0.4 to 0.7, bigger than any in the data)"
KICKS = "with big starting pushes (the floats start three times as far from rest)"
SCALE = "On this scale 0 is the recording controller and 100 is the stall-aware controller."
SIT = {"Recorded conditions": REC, "Strong current": STRONG, "Big waves": WAVES, "Big starting pushes": KICKS}

RESULTS_NEW = [
    {"cols": {
        "What the row asks": "Each row is one comparison between critics, and its first cell asks the question that the "
                             "comparison answers. The first five rows compare the size of the critic's error, and the last "
                             "two compare its direction.",
        "Seeds 0–4": "This gives the lowest and the highest value over the five repeat runs (seeds 0–4). In the first five "
                     "rows the value is an error ratio (×): a critic's typical error divided by that of the critic trained "
                     "on the full log. In the last two rows it is the critic's median extra preference for full throttle, "
                     "in units of return.",
        "Rule": "This is the pass rule fixed in the pre-registration, the plan written down before any result.",
        "Result": "This says whether the rule passed.",
    }},
    {"cols": {
        "Repeat run": "Each row is one repeat run of the harm checks, named by its seed; a seed fixes every random draw "
                      "of a run.",
        "Critic weight 2.5: actor of the full-log critic": "This gives the points in strong current of the actor trained "
            "against the full-log critic (the critic trained on the full log, with nothing removed), at the published "
            "critic weight 2.5. " + SCALE,
        "Critic weight 2.5: actor of the critic trained after the removal": "This gives the points in strong current of "
            "the actor trained against the critic trained after the removal, at the published critic weight 2.5. " + SCALE,
        "Critic weight 5: actor of the full-log critic": "This gives the points in strong current of the actor trained "
            "against the full-log critic, at critic weight 5, the weight at which the check passed. " + SCALE,
        "Critic weight 5: actor of the critic trained after the removal": "This gives the points in strong current of the "
            "actor trained against the critic trained after the removal, at critic weight 5. " + SCALE,
        "Critic weight 5: loss": "This is the number of points lost by trusting the critic trained after the removal: the "
            "points of the actor of the full-log critic minus those of the actor of the critic trained after the removal, "
            "at critic weight 5.",
    }},
    {"cols": {
        "Variant": "Each row is one way of placing extra pull toward the recorded command on the training rows of the "
                   "best-case caution check.",
        "What it is": "This says how the variant sets the dose, the number that multiplies the pull toward the recorded "
                      "command on each training row; a dose of 1 adds nothing.",
        "Why it is here": "This says what the variant is compared for, or what it rules out.",
    }},
    {"cols": dict([("Variant", "This names the variant, the way the extra pull toward the recorded command is placed on "
                               "the training rows; the table above describes each one.")]
                  + [(k, "This gives the points, as a mean over repeat runs (seeds) 0–4, %s. %s" % (v, SCALE))
                     for k, v in SIT.items()])},
    {"cols": {
        "Situation": "Each row is one evaluation bank of 1,000 starts. The recorded conditions are reported but are not "
                     "part of the rule, which looks only at the shifted situations.",
        "Over strength-matched uniform": "This is the points of harm-placed caution minus those of uniform caution of the "
                                         "same strength, as a mean over repeat runs (seeds) 0–4. The rule needs at least "
                                         "+3.",
        "Over amount-matched": "This is the points of harm-placed caution minus those of amount-matched caution, which "
                               "spreads the same total caution evenly. The rule needs at least +3.",
        "Over shuffled": "This is the points of harm-placed caution minus those of the shuffled doses, which are the same "
                         "doses in a random row order. The rule needs at least +3.",
        "Result": "This says whether harm-placed caution reached the pass mark of +3 points against all three controls in "
                  "this situation.",
    }},
    {"cols": {
        "Check: what it guards against": "Each row is one of the nine world checks, which test the world itself and need "
                                         "no learning. The cell names the check and the problem it guards against.",
        "Rule (from the pre-registration)": "This is the check's pass rule, from the pre-registration (the plan written "
                                            "down before any result).",
        "Seed 900": "This is the value on design seed 900, the seed whose recorded log this page uses. A seed fixes every "
                    "random draw of a run, and the design seeds are recordings made only to design and check the world.",
        "Seeds 900–904, lowest–highest": "This gives the lowest and the highest value over design seeds 900–904, five "
                                         "recordings made only to design and check the world, and every check passed "
                                         "on every seed.",
    }},
]


def results(old):
    """New entries for the new tables, then the six kept tables' entries (old entries 5 to 10), unchanged."""
    kept = old[5:]
    assert "Arm" in kept[0]["cols"] and "Strong current: share of states past 80%" in kept[0]["cols"]
    assert "Mark" in kept[3]["cols"] and "Rule" in kept[4]["cols"] and "Item" in kept[5]["cols"]
    return RESULTS_NEW + kept


QUALITIES = {"cols": {
    "Quality": "Each row is one data quality: the same environment and shift, recorded at one level of skill and named "
               "as in the D4RL collection of recorded datasets.",
    "How its recorded commands are made": "This says how often the recording controller's command is replaced by a "
                                          "uniform random command in this quality.",
    "The data": "This gives the points of the recording itself, replayed on the strong test. On this test's scale 0 is "
                "uniform random commands and 100 is the stall-aware controller.",
    "TD3+BC's actor": "This gives the points on the strong test of the actor that TD3+BC learned from the log after the "
                      "removal, as a mean over seeds 0–4. On this test's scale 0 is uniform random commands and 100 is "
                      "the stall-aware controller.",
    "What the removal costs": "This is the points of the actor learned from the full log minus those of the actor "
                              "learned after the removal, on the strong test: how much the missing hold costs the "
                              "learner.",
}}
EXPLORE = {"cols": {
    "Quality": "Each row is one data quality, on which BCA itself is being trained in the exploratory run.",
    "Capped pull: exact weights, each row at its own threshold, minus unweighted": "This gives the return on the strong "
        "test of BCA with its capped pull and exact weights, each training row at its own threshold, minus that of "
        "unweighted BCA, in units of return. Until the run finishes, the cell says running.",
    "Lower edge: exact weights, step by step, minus unweighted": "This gives the return on the strong test of the lower "
        "edge with exact weights, step by step, minus that of the unweighted lower edge, in units of return. Until the "
        "run finishes, the cell says running.",
}}
MARKS = {"group": "Part", "cols": {
    "Check": "Each row is one pass mark of the weighting test, judged for each data quality on its own, and the heading "
             "rows sort the marks into groups.",
    "Passes when": "This is the rule, fixed in version 5 of the specification before any run.",
    "Why this line": "This says why the mark sits where it does.",
    "Result on the five qualities": "This says on which data qualities the mark passed or failed, with the measured "
                                    "numbers; a mark that was not allowed to run says so.",
}}


def weighting(old):
    """The two new tables of "What the results mean", the kept table of the known shift, then the new pass marks."""
    assert len(old) == 2 and "Test setting" in old[0]["cols"] and "Status" in old[1]["cols"]
    return [QUALITIES, EXPLORE, old[0], MARKS]

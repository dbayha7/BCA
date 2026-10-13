"""EXPLORATORY: unit test of run_standins.compute_marks on SYNTHETIC numbers (no data, no training, no return; the
room's outputs are not read). Checks each of plan 4.2's three conditions at and around its boundary, the
every-seed rule, the no-stall loss, the selectivity, the comparison with the range hinge, and the user's choice of
mark forms (spec change C3).

Usage: python test_marks.py
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_standins as R  # noqa: E402

S = R.S
SIT = R.SITUATIONS
SEEDS = (0, 1, 2, 3, 4)


def pts(sc, rec=0.0, other=0.0):
    return {s: (sc if s == "strong_current" else rec if s == "recorded" else other) for s in SIT}


def seed(fam_sc, fam_rec, ns_fam_sc, none_sc=-14.0, none_rec=85.0, pb_sc=70.0, even_sc=-10.0, ns_none_sc=80.0):
    return dict(own_stall={f: pts(fam_sc[f], fam_rec[f]) for f in fam_sc}, own_none_stall=pts(none_sc, none_rec),
                own_no_stall={f: pts(ns_fam_sc[f]) for f in ns_fam_sc}, own_none_no_stall=pts(ns_none_sc),
                room_pb=pts(pb_sc), room_none=pts(none_sc), room_even=pts(even_sc))


def main():
    F = S.FAMILIES
    # perfect band gain 84 (70 - (-14)); half = 42. even pull at -10: condition 3 needs stand-in >= 5.
    sc = {f: 28.0 for f in F}            # gain 42.0: exactly half -> passes (>=)
    sc["density"] = 27.9                 # gain 41.9 -> fails c1
    sc["distance"] = 4.9                 # fails c1 and c3
    rec = {f: 82.0 for f in F}           # 85 - 3: exactly the bound -> passes
    rec["two_head"] = 81.9               # fails c2
    ns = {f: 70.0 for f in F}            # no-stall loss 10
    ns["range_hinge"] = 60.0             # loss 20
    per = {r: seed(sc, rec, ns) for r in SEEDS}
    m = R.compute_marks(per, seeds=SEEDS)
    assert m["range_hinge"]["captures_the_room"], m["range_hinge"]
    assert m["classifier_ratio"]["captures_the_room"]
    assert not m["density"]["captures_the_room"] and not m["density"]["c1_gain_at_least_half_of_perfect_band"]
    assert not m["distance"]["c1_gain_at_least_half_of_perfect_band"] and not m["distance"]["c3_beats_even_pull_by_15"]
    assert m["distance"]["c2_recorded_at_least_none_minus_3"]
    assert not m["two_head"]["c2_recorded_at_least_none_minus_3"] and m["two_head"]["c1_gain_at_least_half_of_perfect_band"]
    assert abs(m["range_hinge"]["gain_strong_current"]["mean"] - 42.0) < 1e-12
    assert abs(m["range_hinge"]["no_stall_loss_strong_current"]["mean"] - 20.0) < 1e-12
    assert abs(m["range_hinge"]["selectivity"]["mean"] - 22.0) < 1e-12
    assert abs(m["classifier_ratio"]["selectivity"]["mean"] - 32.0) < 1e-12
    assert abs(m["classifier_ratio"]["selectivity_minus_range_hinge"]["mean"] - 10.0) < 1e-12
    assert abs(m["range_hinge"]["minus_matched_even_pull_strong_current"]["mean"] - 38.0) < 1e-12
    # pooled means of paired differences: one bad seed can be offset by the others
    per2 = {r: seed(dict(sc, density=28.0 + (5.0 if r < 4 else -20.0)), rec, ns) for r in SEEDS}
    m2 = R.compute_marks(per2, seeds=SEEDS)
    assert m2["density"]["c1_gain_at_least_half_of_perfect_band"]
    # every seed is needed: a missing stand-in (unmatched) on one seed fails every condition
    per3 = {r: seed(sc, rec, ns) for r in SEEDS}
    del per3[3]["own_stall"]["range_hinge"]
    m3 = R.compute_marks(per3, seeds=SEEDS)
    assert not m3["range_hinge"]["captures_the_room"] and m3["range_hinge"]["missing"]
    assert m3["classifier_ratio"]["captures_the_room"]
    # a seed missing from the room (perfect band) fails c1 and c3 of every family
    per4 = {r: seed(sc, rec, ns) for r in SEEDS}
    per4[1]["room_pb"] = None
    per4[1]["room_even"] = None
    m4 = R.compute_marks(per4, seeds=SEEDS)
    assert not any(m4[f]["captures_the_room"] for f in F)
    # the literal forms are labelled reported
    assert all(m[f]["role"].startswith("the plan's literal") for f in S.REPORTED_FAMILIES)
    # the user's choice (approval.json, spec change C3) sets which forms carry the mark; the numbers do not change
    lit = S.MARK_FORMS["literal"]["mark"]
    m5 = R.compute_marks(per, seeds=SEEDS, mark_families=lit)
    assert all(m5[f]["role"] == "stand-in (mark)" for f in lit)
    assert all(m5[f]["role"].startswith("re-referenced") for f in ("density", "distance", "two_head"))
    assert all(m5[f]["captures_the_room"] == m[f]["captures_the_room"] for f in F)
    assert set(S.MARK_FORMS["re_referenced"]["mark"]) | set(S.MARK_FORMS["re_referenced"]["reported"]) == set(F)
    assert set(lit) | set(S.MARK_FORMS["literal"]["reported"]) == set(F) and len(lit) == 5
    # the chat message's choice phrase: exactly one of the two
    assert S.choice_in_message("abc approve F1-F3 xyz") == "re_referenced"
    assert S.choice_in_message("Decline f1-f3 please") == "literal"
    assert S.choice_in_message("approve F1-F3 or decline F1-F3") is None
    assert S.choice_in_message("yes, go ahead") is None
    print("test_marks: OK")


if __name__ == "__main__":
    main()

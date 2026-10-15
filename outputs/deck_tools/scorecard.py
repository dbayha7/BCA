# ------------------------------------------------------------------ scorecard (after the roadmap)
sc = clone(nx)                                     # the NEXT slide, already cut to four cards
set_text(sc, "Text 0", "SCORECARD")
set_text(sc, "Text 1", "The guarantee is sounder; there is no win for learning yet")
cards4 = [("Studies 1–5", "Calibration: flaws found, fixed",
           "Four real flaws in how BCA sets its cut-off: wrong method, blind to shift, related rows in its bank, the "
           "wrong critic copy. Each one now has a fix, though it isn't installed in BCA everywhere yet."),
          ("No study yet", "BCA has not improved a policy",
           "Studies 1–5 used frozen critics, so none measured learning. The only return number is one seed on one "
           "dataset, and it is descriptive only."),
          ("Studies 6–7", "Today's channel can't reach the harm",
           "BCA adds extra pull toward the logged data. Even placed perfectly, that pull wins back almost nothing of "
           "the 99 points the data gap costs."),
          ("Study 8", "A channel that works; BCA not yet",
           "Penalising the value of the actor's own command wins back 94 of 99 points with perfect information. "
           "BCA's band failed its first test; a weighted version is running.")]
for k, (num, head, body) in enumerate(cards4):
    set_text(sc, f"Text {3 + 4 * k}", num)
    set_text(sc, f"Text {4 + 4 * k}", head)
    set_text(sc, f"Text {5 + 4 * k}", body)
    o = shp(sc, f"Text {3 + 4 * k}")
    for p in o.text_frame.paragraphs:
        for r in p.runs:
            r.font.size = Pt(15)
    shp(sc, f"Text {4 + 4 * k}").height = Inches(0.6)
    b = shp(sc, f"Text {5 + 4 * k}"); b.top = Inches(3.05); b.height = Inches(1.6)
    shp(sc, f"Text {4 + 4 * k}").top = Inches(2.35)
    for p in b.text_frame.paragraphs:
        for r in p.runs:
            r.font.size = Pt(12.5)
set_text(sc, "Text 22", "In one sentence: studies 1–5 made the guarantee trustworthy; studies 6–8 asked whether it helps "
                        "learning, and so far it does not in its current form.")
shp(sc, "Text 22").width = Inches(11.56)
REPLACE[19] = [road_a, road_b, sc]

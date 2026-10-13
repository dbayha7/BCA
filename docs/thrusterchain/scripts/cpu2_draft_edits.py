"""CPU 2: small asserted wording fixes in the two drafts (notes only, not the page)."""
P1 = "/mnt/c/Users/David/dev/bca-shift-work/page/notes/results_draft.html"
P2 = "/mnt/c/Users/David/dev/bca-shift-work/page/notes/weighting_results_draft.html"
def fix(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new, n in pairs:
        assert s.count(old) == n, (path, old, s.count(old))
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)
fix(P1, [
    ("is the error just from having less data?", "is the error only from having less data?", 2),
])
fix(P2, [
    ('<span class="t">Data, learners and room checks</span>', '<span class="t">Data, learners and checks without BCA</span>', 1),
    ("(measured on the room check's actors), that predicts", "(measured on the actors of this room check), that predicts", 1),
    ("times the return per unit of dose measured on the room check's actors</td>", "times the return per unit of dose measured on the actors of the room check above</td>", 1),
    ("the random share falls from every command to 65% over the episodes, like a learning curve", "the random share runs from every command down to 65% across its episodes, like a learning curve", 1),
    ("This matches, on different data, what the pre-registered test above found: a pull toward the recorded\n      commands has little room, while a penalty on the value of the actor's own command, given perfect information,\n      has a lot.",
     "This matches, on different data, what the pre-registered test above and its follow-up (Stage 1) found: a pull\n      toward the recorded commands has little room, while a penalty on the value of the actor's own command, given\n      perfect information, has a lot.", 1),
])
print("ok")

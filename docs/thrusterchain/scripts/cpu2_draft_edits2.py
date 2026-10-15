"""CPU 2: remove nested comment markers from the drafts' header comments (notes only)."""
P1 = "/mnt/c/Users/David/dev/bca-shift-work/page/notes/results_draft.html"
P2 = "/mnt/c/Users/David/dev/bca-shift-work/page/notes/weighting_results_draft.html"
def fix(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new, n in pairs:
        assert s.count(old) == n, (path, old, s.count(old))
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)
fix(P1, [("   - <!-- nums: ... --> after a block lists", "   - A comment starting \"nums:\" after a block lists", 1),
         ("   - <!-- KEEP: ... --> marks a block", "   - A comment starting \"KEEP:\" marks a block", 1)])
fix(P2, [("Every number is listed in a <!-- nums --> comment with its", "Every number is listed in a comment starting \"nums:\" with its", 1)])
print("ok")

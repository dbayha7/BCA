"""Manual-review helper for the rerun (read-only): lists every distinct note under a cell tuple and how often it appears,
every tuple that uses a page nickname together with whether its note explains it, the records whose numbers are not in
the page text, and the deliberately kept items. Reads notes/tooltips_inventory_after.json."""
import collections
import json
import re

INV = "/mnt/c/Users/David/dev/bca-shift-work/page/notes/tooltips_inventory_after.json"
d = json.load(open(INV, encoding="utf-8"))
recs = d["records"]

notes = collections.Counter()
for r in recs:
    if r.get("note"):
        for s in re.split(r"(?<=\.)\s+(?=[A-Z])", r["note"]):
            notes[s] += 1
print("== distinct note sentences under cell tuples")
for s, n in notes.most_common():
    print(f"{n:4d}  {s}")

NICK = {"oracle": r"\boracle", "gate": r"\bgate\b|gate's", "gapped critic": r"gapped critic", "room check": r"room check",
        "futility look": r"futility look", "marks look": r"marks look", "approval": r"approval (file|program)",
        "choice message": r"choice message", "stand-in": r"stand-ins?\b", "pins": r"\bpins?\b", "Stage": r"Stage [12]",
        "the stop": r"\bthe stop\b", "main arm": r"main arm", "step-up": r"step-up", "manifest": r"manifest",
        "lock": r"\block\b", "hash": r"\bhash", "pre-registration": r"pre-regist", "seed": r"\bseeds?\b"}
print("\n== tuples that use a page nickname, and whether the tuple explains it")
for name, pat in NICK.items():
    hits = [r for r in recs if r["kind"].startswith("cell tuple") and re.search(pat, r["text"], re.I)]
    unexplained = [r["id"] for r in hits if not re.search(pat, r.get("note", ""), re.I)
                   and not any(re.search(pat, c["description"], re.I) for c in r["cells"])]
    print(f"{name:18s} {len(hits):3d} tuples; not explained in their note or column description: {len(unexplained)} "
          f"{unexplained[:12]}")

print("\n== numbers not found in the page text")
for r in recs:
    if r.get("numbers_not_found_in_page_text"):
        print(f"{r['id']:14s} {r['kind']:28s} {r['numbers_not_found_in_page_text']}  {r['text'][:90]!r}")

print("\n== deliberately kept")
for r in recs:
    for k in r.get("kept", []):
        print(f"{r['id']:14s} {k['type']:10s} {k['detail'][:70]}")

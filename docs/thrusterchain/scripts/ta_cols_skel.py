"""Write tmp/ta/cols_skeleton.txt: every distinct column description in the page's COLS object, with where it is used."""
import json, collections
P = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"
s = open(P, encoding="utf-8").read()
head = "  const COLS = "
a = s.index(head); z = s.index(";\n", a)
cols = json.loads(s[a + len(head):z].replace("<\/", "</"))
seen = collections.OrderedDict()
for sec, lst in cols.items():
    for i, e in enumerate(lst):
        for name, v in e["cols"].items():
            d = v[1] if isinstance(v, list) else v
            seen.setdefault(d, []).append(f"{sec}[{i}] {name!r}" + (f" -> shown as {v[0]!r}" if isinstance(v, list) else ""))
        if e.get("group"):
            print("group", sec, i, e["group"])
with open("/mnt/c/Users/David/dev/bca-shift-work/page/tmp/ta/cols_skeleton.txt", "w", encoding="utf-8") as f:
    for d, w in seen.items():
        f.write("# " + " | ".join(w)[:300] + "\n")
        f.write(repr(d) + "\n\n")
print(len(seen), "distinct descriptions;", sum(len(e["cols"]) for l in cols.values() for e in l), "entries")

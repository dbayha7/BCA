"""List the inventory's records by kind into tmp/ta/list_<kind>.txt for reading."""
import json, collections, os, re
P = "/mnt/c/Users/David/dev/bca-shift-work/page"
d = json.load(open(P + "/notes/tooltips_inventory.json", encoding="utf-8"))
by = collections.defaultdict(list)
for r in d["records"]:
    by[r["kind"]].append(r)
for k, lst in by.items():
    name = re.sub(r"\W+", "_", k)[:40]
    with open(f"{P}/tmp/ta/list_{name}.txt", "w", encoding="utf-8") as f:
        for r in lst:
            f.write(f"== {r['id']} x{r['count']} | {r['where'][0][:110]}\n")
            f.write(f"   GEN: {r.get('generator','')[:160]}\n")
            if r["title"]:
                f.write(f"   TITLE: {r['title']}\n")
            f.write(f"   TEXT: {r['text']}\n")
            for p in r["problems"]:
                f.write(f"     - {p['type']}: {p['detail'][:200]}\n")
    print(k, len(lst), name)

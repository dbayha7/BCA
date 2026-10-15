"""For every hand-written data-tip record in the inventory, count its attribute form in the page (read-only)."""
import json, html as H, collections
P = "/mnt/c/Users/David/dev/bca-shift-work/page"
s = open(P + "/thrusterchain.html", encoding="utf-8").read()
inv = json.load(open(P + "/notes/tooltips_inventory.json", encoding="utf-8"))["records"]
esc = lambda x: H.escape(x, quote=True)
bad = 0
kinds = collections.Counter()
for r in inv:
    if not r["id"].startswith("tip-") or r["kind"] in ("formula legend", "controller guide", "chart band"):
        continue
    kinds[r["kind"]] += 1
    t, x = r["title"], r["text"]
    pair = f'data-tip-title="{esc(t)}" data-tip="{esc(x)}"' if t else None
    n_pair = s.count(pair) if pair else None
    n_tip = s.count(f'data-tip="{esc(x)}"')
    if (pair and n_pair != r["count"]) or (not pair and n_tip != r["count"]):
        bad += 1
        print(r["id"], r["kind"], "count", r["count"], "pair", n_pair, "tip", n_tip, repr(t), repr(x[:60]))
print("checked", sum(kinds.values()), dict(kinds), "mismatches", bad)

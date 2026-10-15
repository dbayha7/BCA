"""Read-only: parse the live panel's payload (const D = {...};) and describe its structure.
Also writes the stall|recorded|stall_aware episode and its water to notes/demo_episode.json for the prototype test."""
import json

P = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"
OUT = "/mnt/c/Users/David/dev/bca-shift-work/page/notes/demo_episode.json"
s = open(P, encoding="utf-8").read()
start = s.index("const D = ")
end = s.index("\n", start)
line = s[start:end]
assert line.endswith(";"), line[-50:]
D = json.loads(line[len("const D = "):-1])
print("top keys", list(D.keys()))


def desc(v, depth=0, maxdepth=4):
    pad = "  " * depth
    if isinstance(v, dict):
        ks = list(v.keys())
        print(pad + f"dict[{len(ks)}] keys: {ks[:12]}{' ...' if len(ks) > 12 else ''}")
        if depth < maxdepth:
            for k in ks[:3]:
                print(pad + f" [{k!r}]:")
                desc(v[k], depth + 1, maxdepth)
    elif isinstance(v, list):
        inner = type(v[0]).__name__ if v else "-"
        print(pad + f"list[{len(v)}] of {inner}; first: {json.dumps(v[0])[:160] if v else ''}")
    else:
        print(pad + f"{type(v).__name__}: {json.dumps(v)[:160]}")


for k in D:
    print("=====", k)
    desc(D[k], 1, 3)

R = D["roll"]["stall|recorded|stall_aware"]
print("===== stall|recorded|stall_aware fields")
for k, v in R.items():
    if isinstance(v, list):
        print(k, len(v), json.dumps(v[:3])[:200], "...", json.dumps(v[-1])[:120])
    else:
        print(k, json.dumps(v)[:400])
W = D["water"]["recorded"]
print("===== water.recorded fields")
for k, v in W.items():
    if isinstance(v, list):
        print(k, len(v), json.dumps(v[:5])[:200])
    else:
        print(k, json.dumps(v))
print("ref stall recorded stall_aware", D["ref"]["stall"]["recorded"]["stall_aware"])
print("trigger", D.get("trigger"))
a1 = [row[0] for row in R["a"]]
f1 = [row[0] for row in R["f"]]
print("thruster1 throttle min/max", min(a1), max(a1), "push min/max", min(f1), max(f1))
allf = [x for row in R["f"] for x in row]
print("all push min/max", min(allf), max(allf))
pos = [R["s"][t][2 * j] for t in range(len(R["s"])) for j in range(5)]
print("positions min/max", min(pos), max(pos))
print("steps with thruster1 at cap |a|>=0.799:", sum(1 for x in a1 if abs(x) >= 0.799))
print("stalled strings sample", R["st"][:5])
print("len s", len(R["s"]), "len a", len(R["a"]), "len r", len(R["r"]), "len g", len(R["g"]))
json.dump({"roll": R, "water": W, "ref": D["ref"]["stall"]["recorded"]["stall_aware"], "trigger": D.get("trigger")},
          open(OUT, "w"), separators=(",", ":"))
print("wrote", OUT)

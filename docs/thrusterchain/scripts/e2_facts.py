"""Read-only: facts of the demo episode (stall-aware controller, stalling world, recorded conditions, start D.start),
counted the way the live panel counts states: state 0 is the start, state 99 the last state in which the controller acts.
usage: python3 e2_facts.py [PAGE]"""
import json
import math
import sys

PAGE = sys.argv[1] if len(sys.argv) > 1 else "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"
s = open(PAGE, encoding="utf-8").read()
i = s.index("\nconst D = ") + len("\nconst D = ")
j = s.index("\n", i)
D = json.loads(s[i:j].rstrip().rstrip(";"))
R = D["roll"]["stall|recorded|stall_aware"]
W = D["water"]["recorded"]
N = len(R["a"])
print("keys", list(D.keys()), "start", D["start"], "trigger", D["trigger"], "steps", N, "states", len(R["s"]))
print("current c", W["c"], "amp", W["amp"], "dist", W["dist"])
print("summary", json.dumps(R["sum"]))
print("ref recorded stall_aware", D["ref"]["stall"]["recorded"]["stall_aware"])
print("return g[99]", R["g"][-1], " r[0]", R["r"][0])
pos = [[x[2 * k] for k in range(5)] for x in R["s"]]
dist = [[math.hypot(x[2 * k], x[2 * k + 1]) for k in range(5)] for x in R["s"]]
maxp = [max(abs(v) for v in row) for row in pos]
maxd = [max(row) for row in dist]
for th in (0.3, 0.2):
    t1 = next(t for t in range(N) if all(m <= th for m in maxp[t:N]))
    t2 = next(t for t in range(N) if all(m <= th for m in maxd[t:N]))
    print(f"every float position within {th} of rest from state {t1} on; distance (pos,speed) within {th} from state {t2} on")
cap = [t for t in range(N) if any(abs(abs(a) - 0.8) < 1e-6 for a in R["a"][t])]
print("states with some throttle at the 80% cap:", cap, [[k + 1 for k in range(4) if abs(abs(R['a'][t][k]) - 0.8) < 1e-6] for t in cap])
print("any past 80%:", any(abs(a) > 0.8 + 1e-9 for row in R["a"] for a in row), "stalled:", sum(row.count("1") for row in R["st"]))
for t0 in (8, 9, 10, 11, 12):
    a1 = [abs(R["a"][t][0]) for t in range(t0, N)]
    print(f"from state {t0}: thruster 1 throttle {min(a1):.3f}..{max(a1):.3f}")
push = W["push"]
print("push range", min(push), max(push), "above trigger", sum(p > D["trigger"] for p in push))
a1 = [R["a"][t][0] for t in range(N)]
m = lambda xs: sum(xs) / len(xs)
for t0 in (10,):
    xs, ys = push[t0:], a1[t0:]
    mx, my = m(xs), m(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    r = cov / math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    print(f"correlation of push and thruster 1 throttle from state {t0}: {r:.3f}")
for t in (0, 1, 2, 3, 10, 40, 99):
    print(f"state {t}: pos {[round(v, 3) for v in pos[t]]} a {R['a'][t]} f {R['f'][t]} push {push[t]} g {R['g'][t]}")
print("float cost state 0", sum(v * v for v in R["s"][0][:10]))

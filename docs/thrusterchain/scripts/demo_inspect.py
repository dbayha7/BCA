"""Read-only inspection of thrusterchain.html for the demo-episode plan. Writes nothing to the page."""
import re, sys, json

P = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"
s = open(P, encoding="utf-8").read()
print("chars", len(s))
lines = s.split("\n")
print("lines", len(lines))
for i, l in enumerate(lines):
    if len(l) > 5000:
        print("long line", i + 1, len(l), repr(l[:200]))

def show(pat, width=300, maxn=20):
    n = 0
    for m in re.finditer(pat, s):
        a = max(0, m.start() - width)
        b = min(len(s), m.end() + width)
        print("----", pat, "@", m.start())
        print(s[a:b])
        n += 1
        if n >= maxn:
            break
    print("==== count", pat, len(re.findall(pat, s)))

mode = sys.argv[1] if len(sys.argv) > 1 else "overview"
if mode == "overview":
    for pat in [r"task-diagram", r'id="sim"', r"<canvas", r"<script", r"</script>", r"stall\|recorded\|stall_aware", r"__DATA__"]:
        print("====", pat, [m.start() for m in re.finditer(pat, s)][:30])
elif mode == "show":
    show(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 300, int(sys.argv[4]) if len(sys.argv) > 4 else 20)
elif mode == "range":
    a, b = int(sys.argv[2]), int(sys.argv[3])
    print(s[a:b])

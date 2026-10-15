"""Read-only survey of thrusterchain.html structure for the structure plan.

Usage: survey_structure.py outline | range A B | grep REGEX | chars A B
Long lines are summarized; MathJax containers are shown as [M:label].
"""
import re, sys

P = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"
src = open(P, encoding="utf-8").read()
lines = src.split("\n")


def strip_mjx(s):
    return re.sub(r"<mjx-container[^>]*?aria-label=\"([^\"]*)\".*?</mjx-container>",
                  lambda m: "[M:" + m.group(1) + "]", s, flags=re.S)


mode = sys.argv[1] if len(sys.argv) > 1 else "outline"

if mode == "outline":
    print("bytes", len(src.encode("utf-8")), "chars", len(src), "lines", len(lines))
    print("\n== tags with id (in order) ==")
    for m in re.finditer(r"<(section|header|nav|aside|div|figure|details|h1|h2|h3|table|canvas|svg|button|input|select|main|footer|p|span|ol|ul|a)\b([^>]*?)\bid=\"([^\"]+)\"", src):
        ln = src.count("\n", 0, m.start()) + 1
        print(ln, m.start(), m.group(1), m.group(3))
elif mode == "range":
    a, b = int(sys.argv[2]), int(sys.argv[3])
    lim = int(sys.argv[4]) if len(sys.argv) > 4 else 600
    for i in range(a - 1, min(b, len(lines))):
        l2 = strip_mjx(lines[i])
        if len(l2) > lim:
            l2 = l2[:lim] + f" ...[{len(l2)} chars]"
        print(f"{i+1}: {l2}")
elif mode == "grep":
    pat = re.compile(sys.argv[2])
    w = int(sys.argv[3]) if len(sys.argv) > 3 else 150
    for i, l in enumerate(lines):
        if len(l) > 100000:
            continue
        for m in pat.finditer(l):
            s = max(0, m.start() - w)
            e = min(len(l), m.end() + w)
            print(f"{i+1}:{m.start()}: {strip_mjx(l[s:e])}")
elif mode == "chars":
    a, b = int(sys.argv[2]), int(sys.argv[3])
    print(strip_mjx(src[a:b]))

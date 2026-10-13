"""Read-only: print raw source around given markers (count and context). Usage: s1_peek.py WIDTH marker1 [marker2 ...]"""
import sys
P = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"
s = open(P, encoding="utf-8").read()
w = int(sys.argv[1])
for m in sys.argv[2:]:
    n = s.count(m)
    i = s.find(m)
    print(f"===== [{m[:60]}] count={n} at {i} line {s.count(chr(10), 0, i) + 1}")
    if i >= 0:
        print(repr(s[max(0, i - w): i + len(m) + w]))

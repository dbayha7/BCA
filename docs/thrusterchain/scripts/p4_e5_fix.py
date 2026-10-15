#!/usr/bin/env python3
"""Page pass 4, edit 5, follow-up: one wording found while reading the final screenshots.

"What the results mean" in the weighting test said that no weighted run gained the required 1.8 units of return. One
did gain more in size (random, lower edge: +0.21 points, about 2.7 units), but its lower bound is below 0; the bar fixed
in advance is a gain of 1.8 units WITH a lower bound above 0, as exploratory result B states. Exact, counted replacement.

    python3 scripts/p4_e5_fix.py      (run from the page directory, after p4_e5_apply.py)
"""
import hashlib
import os

PAGE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "thrusterchain.html")
s = open(PAGE, encoding="utf-8").read()
assert hashlib.md5(s.encode("utf-8")).hexdigest() == "1dec7e73299f11499da4292a01de5c07", "not the page p4_e5_apply.py wrote"
a = s.index('<section id="weighting-test"')
b = s.index("</section>", a)
old = """no weighted run
        gained the required 1.8 units of return on any quality."""
new = """no weighted run
        gained the required 1.8 units of return, with a lower bound above 0, on any quality."""
assert s[a:b].count(old) == 1, s[a:b].count(old)
s = s[:a] + s[a:b].replace(old, new) + s[b:]
open(PAGE, "w", encoding="utf-8", newline="").write(s)
print("wrote", PAGE, hashlib.md5(s.encode("utf-8")).hexdigest())

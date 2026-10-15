"""Edit 2, fix 1: the task diagram's legend names the swing label as both drawings print it (the wide drawing says
"swing: 63 steps", the phone-width one "each float's own swing, in steps: 63"). One exact, asserted replacement."""
import hashlib
P = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"
B = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.v9.html"
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
b0 = md5(B)
s = open(P, encoding="utf-8").read()
old = "<dt>swing: 63 steps</dt><dd>A float's period, the steps one swing takes, rounded:"
new = "<dt>swing, in steps</dt><dd>A float's period, the steps one swing takes, rounded under each float:"
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new, 1)
open(P, "w", encoding="utf-8").write(s)
assert md5(B) == b0
print("fixed;", len(s), "bytes, md5", md5(P), "; backup", b0)

"""Read-only: raw source context of a phrase inside one top-level block of <main> (whitespace-tolerant).
Usage: s1_bpeek.py WIDTH block:phrase ..."""
import re, sys
P = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"
s = open(P, encoding="utf-8").read()
m0 = s.find("<main>"); m1 = s.find("</main>")
pat = re.compile(r"\n(  <header class=\"head\" id=\"top\">|  <section id=\"([^\"]+)\">|    <figure class=\"panel\" id=\"sim\")")
st = [(m.start() + 1, m.group(2) or ("top" if "header" in m.group(1) else "sim")) for m in pat.finditer(s, m0, m1)]
blocks = {}
for i, (a, bid) in enumerate(st):
    blocks[bid] = (a, st[i + 1][0] if i + 1 < len(st) else m1)
w = int(sys.argv[1])
for arg in sys.argv[2:]:
    b, ph = arg.split(":", 1)
    a, e = blocks[b]
    rx = re.compile(r"\s+".join(re.escape(x) for x in ph.split()))
    ms = list(rx.finditer(s, a, e))
    print(f"===== {b} / [{ph}] matches in block={len(ms)}")
    for m in ms[:2]:
        print(f"  line {s.count(chr(10), 0, m.start()) + 1}: " + repr(s[max(a, m.start() - w): m.end() + w]))

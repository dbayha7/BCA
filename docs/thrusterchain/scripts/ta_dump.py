"""Dump the page's tooltip scripts, styles and COLS object into tmp/ta/ for reading. Read-only on the page."""
import re, os
P = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"
OUT = "/mnt/c/Users/David/dev/bca-shift-work/page/tmp/ta"
s = open(P, encoding="utf-8").read()
scripts = [(m.start(), m.group(1)) for m in re.finditer(r"<script\b[^>]*>(.*?)</script>", s, flags=re.S)]
print(len(scripts), "scripts")
for i, (pos, body) in enumerate(scripts):
    first = body.strip().split("\n", 1)[0][:100]
    print(i, pos, len(body), repr(first))
styles = [(m.start(), m.group(1)) for m in re.finditer(r"<style\b[^>]*>(.*?)</style>", s, flags=re.S)]
print(len(styles), "styles", [ (p, len(b)) for p, b in styles])
for i, (pos, body) in enumerate(scripts):
    if len(body) < 400000:
        open(os.path.join(OUT, f"script{i:02d}.js"), "w", encoding="utf-8").write(body)
for i, (pos, body) in enumerate(styles):
    open(os.path.join(OUT, f"style{i:02d}.css"), "w", encoding="utf-8").write(body)
lines = s.split("\n")
print("lines", len(lines), "longest", max(len(l) for l in lines))

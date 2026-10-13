"""Print source snippets of the page's tooltip texts: data-tip contexts, legends, guide templates, chart titles."""
import re, sys
P = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"
OUT = "/mnt/c/Users/David/dev/bca-shift-work/page/tmp/ta/src_snippets.txt"
s = open(P, encoding="utf-8").read()
o = []
tips = list(re.finditer(r'data-tip="([^"]*)"', s))
o.append(f"data-tip attributes: {len(tips)}")
for m in tips[:6]:
    o.append("CTX: " + s[max(0, m.start()-200):m.end()+60].replace("\n", "\n"))
o.append("\n\n#### symkey legends")
for m in re.finditer(r'<div class="symkey[^"]*"[^>]*>.*?</dl>', s, flags=re.S):
    o.append(m.group(0))
    o.append("-----")
o.append("\n\n#### templates")
for m in re.finditer(r'<template id="tip-ctrl[^"]*">.*?</template>', s, flags=re.S):
    o.append(m.group(0))
o.append("\n\n#### chart titles")
for m in re.finditer(r'<title>(.*?)</title>', s, flags=re.S):
    o.append(f"{m.start()}: {m.group(1)}")
o.append("\n\n#### data-tip-title + data-tip orders")
both = re.findall(r'data-tip-title="[^"]*"\s+data-tip="', s)
rev = re.findall(r'data-tip="[^"]*"\s+data-tip-title="', s)
o.append(f"title-then-tip {len(both)}, tip-then-title {len(rev)}")
open(OUT, "w", encoding="utf-8").write("\n".join(o))
print(len(o), "items ->", OUT)

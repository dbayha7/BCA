"""Print each match of a regex in the page's visible text (tags stripped) with context. Usage: e7_grep.py REGEX [CTX] [FILE]"""
import re, sys, html
pat = sys.argv[1]; ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 120
f = sys.argv[3] if len(sys.argv) > 3 else "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"
raw = sys.argv[4] == "raw" if len(sys.argv) > 4 else False
s = open(f, encoding="utf-8").read()
if not raw:
    s = s[:s.index("<script")] if "<script" in s else s
    s = re.sub(r"<mjx-container[^>]*aria-label=\"([^\"]*)\"[^>]*>.*?</mjx-container>", r" ⟨\1⟩ ", s, flags=re.S)
    s = re.sub(r"<style.*?</style>", " ", s, flags=re.S)
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s)
n = 0
for m in re.finditer(pat, s):
    n += 1
    print(f"[{n}] …{s[max(0,m.start()-ctx):m.end()+ctx]}…\n")
print("matches:", n)

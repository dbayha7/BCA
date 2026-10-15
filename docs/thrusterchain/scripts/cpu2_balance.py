"""CPU 2: tag balance of the two drafts, comments removed (read only)."""
import re
from html.parser import HTMLParser
VOID = {"br", "img", "input", "meta", "link", "hr", "col", "wbr"}
class B(HTMLParser):
    def __init__(s):
        super().__init__(); s.st = []; s.err = []
    def handle_starttag(s, t, a):
        if t not in VOID: s.st.append((t, s.getpos()))
    def handle_endtag(s, t):
        if t in VOID: return
        if s.st and s.st[-1][0] == t: s.st.pop()
        else: s.err.append((t, s.getpos(), s.st[-1] if s.st else None))
for p in ["results_draft.html", "weighting_results_draft.html"]:
    x = open("/mnt/c/Users/David/dev/bca-shift-work/page/notes/" + p, encoding="utf-8").read()
    x = re.sub(r"<!--.*?-->", "", x, flags=re.S)
    b = B(); b.feed(x)
    print(p, "unclosed:", b.st[:5], "errors:", b.err[:5])

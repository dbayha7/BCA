"""Read-only: top-level children of <main>, every in-page link and its target, and first uses of a few terms."""
import re
from html.parser import HTMLParser
from collections import Counter, defaultdict

P = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"
src = open(P, encoding="utf-8").read()
# cut the scripts so the data payload does not slow the parser
body = src.split("<script>")[0]

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr",
        "path", "line", "rect", "circle", "polygon", "polyline", "use", "ellipse", "stop"}


class P1(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.main_children = []
        self.ids = {}
        self.section_of = {}
        self.cur_top = None
        self.links = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if self.stack and self.stack[-1] == "main" and len([s for s in self.stack if s == "main"]) == 1 and self.stack[-1] == "main":
            self.main_children.append((tag, a.get("id"), a.get("class"), self.getpos()[0]))
            self.cur_top = a.get("id") or tag
        if "id" in a:
            self.ids[a["id"]] = (tag, self.getpos()[0], self.cur_top)
        if tag == "a" and a.get("href", "").startswith("#"):
            self.links.append((a["href"][1:], self.getpos()[0], self.cur_top))
        if tag not in VOID and not tag.startswith("mjx") and not (tag in ("svg",) and False):
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids[a["id"]] = (tag, self.getpos()[0], self.cur_top)

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        # pop to matching tag
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i] == tag:
                del self.stack[i:]
                break


p = P1()
p.feed(body)
print("== main children ==")
for c in p.main_children:
    print(c)
print("\n== in-page links: target -> [from top-level block] ==")
by = defaultdict(list)
for tgt, ln, top in p.links:
    by[tgt].append(f"{top}@{ln}")
for tgt in by:
    t = p.ids.get(tgt)
    print(f"#{tgt:24s} target_in={t[2] if t else 'MISSING'}  from: {', '.join(by[tgt])}")

print("\n== first uses ==")
txt = re.sub(r"<mjx-container.*?</mjx-container>", " ", body, flags=re.S)
for term in ["Gymnasium", "Box(", "evaluation bank", "start law", "time step", "gate", "points", "s_t", "float32"]:
    i = txt.find(term)
    ln = txt.count("\n", 0, i) + 1 if i >= 0 else None
    print(term, ln, repr(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", txt[max(0, i - 80): i + 120]))) if i >= 0 else "")

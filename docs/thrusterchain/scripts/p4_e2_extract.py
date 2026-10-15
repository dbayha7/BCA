"""Read-only: extract the current markup and text of the draft-A targets.

Writes tmp/p4/draftA/<id>.html (markup with MathJax collapsed to its aria-label and inline SVG
collapsed to a stub) and tmp/p4/draftA/<id>.txt (reader-visible text, one block per line).
The page itself is never written.
"""
import hashlib
import os
import re
import sys

from html.parser import HTMLParser

# Text extractor copied from e3_extract.py (importing that module runs its main and writes notes files).
BLOCK = {"p", "li", "h1", "h2", "h3", "h4", "h5", "figcaption", "caption", "dt", "dd",
         "summary", "button", "label", "blockquote", "pre", "div", "text", "title", "option",
         "legend", "tr", "nav", "header", "footer", "section", "figure", "aside", "details", "table",
         "ul", "ol", "dl", "main", "svg", "g"}
VOID = {"br", "img", "input", "meta", "link", "hr", "source", "wbr", "path", "rect", "circle", "line",
        "polyline", "polygon", "ellipse", "use", "stop"}


class P(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []  # (tag, id)
        self.skip = 0  # inside script/style
        self.mjx = 0
        self.buf = []
        self.lines = []
        self.section = "(top)"

    def where(self):
        sec = self.section
        near = ""
        for tag, i in reversed(self.stack):
            if i and not i.startswith("MJX"):
                near = i
                break
        return sec, near

    def flush(self):
        t = re.sub(r"\s+", " ", "".join(self.buf)).strip()
        self.buf = []
        if t:
            sec, near = self.where()
            self.lines.append((sec, near, t))

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if self.mjx:
            if tag == "mjx-container":
                self.mjx += 1
            return
        if tag in ("script", "style"):
            self.skip += 1
            self.stack.append((tag, None))
            return
        if tag == "mjx-container":
            self.buf.append(" ⟦math: " + a.get("aria-label", "?") + "⟧ ")
            self.stack.append((tag, None))
            self.mjx = 1
            return
        if tag in BLOCK:
            self.flush()
        if tag in ("td", "th"):
            self.buf.append(" ¦ ")
        if tag == "section" and a.get("id"):
            self.section = a["id"]
        # reader-visible attributes on controls
        for k in ("aria-label", "title", "placeholder", "alt", "data-tip"):
            if a.get(k) and tag != "mjx-container":
                sec, near = self.where()
                self.lines.append((sec, a.get("id") or near, f"<{tag} @{k}> {a[k]}"))
        if tag not in VOID:
            self.stack.append((tag, a.get("id")))
        elif tag == "br":
            self.buf.append(" ")

    def handle_startendtag(self, tag, attrs):
        a = dict(attrs)
        if self.mjx or self.skip:
            return
        for k in ("aria-label", "title", "placeholder", "alt"):
            if a.get(k):
                sec, near = self.where()
                self.lines.append((sec, a.get("id") or near, f"<{tag} @{k}> {a[k]}"))

    def handle_endtag(self, tag):
        if self.mjx:
            if tag == "mjx-container":
                self.mjx -= 1
                if self.mjx == 0 and self.stack and self.stack[-1][0] == "mjx-container":
                    self.stack.pop()
            return
        if tag in ("script", "style"):
            self.skip = max(0, self.skip - 1)
            if self.stack and self.stack[-1][0] == tag:
                self.stack.pop()
            return
        if tag in BLOCK:
            self.flush()
        # pop to the matching tag
        for j in range(len(self.stack) - 1, -1, -1):
            if self.stack[j][0] == tag:
                del self.stack[j:]
                break

    def handle_data(self, data):
        if self.skip or self.mjx:
            return
        self.buf.append(data)



PAGE = sys.argv[2] if len(sys.argv) > 2 else "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/c/Users/David/dev/bca-shift-work/page/tmp/p4/edit2/cur"
os.makedirs(OUT, exist_ok=True)
s = open(PAGE, encoding="utf-8").read()
md5_before = hashlib.md5(s.encode("utf-8")).hexdigest()

starts = {
    "top": '<header class="head" id="top">',
    "action-space": '<section id="action-space"',
    "observation-space": '<section id="observation-space"',
    "starting-state": '<section id="starting-state"',
    "dynamics": '<section id="dynamics"',
    "rewards": '<section id="rewards"',
    "episode-end": '<section id="episode-end"',
    "sim": '<figure class="panel" id="sim"',
    "task": '<section id="task"',
    "description": '<section id="description"',
    "recording-controller": '<section id="recording-controller"',
    "reference-scores": '<section id="reference-scores"',
    "dataset": '<section id="dataset"',
    "shift": '<section id="shift"',
    "critic-in-action": '<section id="critic-in-action"',
    "methods": '<section id="methods"',
    "harm-checks": '<section id="harm-checks"',
    "bca-test": '<section id="bca-test"',
    "results": '<section id="results"',
    "weighting-test": '<section id="weighting-test"',
    "arguments": '<section id="arguments"',
    "code-names": '<section id="code-names"',
    "END": '</main>',
}
pos = {}
for k, m in starts.items():
    assert s.count(m) == 1, (k, s.count(m))
    pos[k] = s.index(m)
keys = list(starts)
assert [pos[k] for k in keys] == sorted(pos[k] for k in keys)


def collapse(h):
    h = re.sub(r'<mjx-container[^>]*aria-label="([^"]*)"[^>]*>.*?</mjx-container>',
               lambda m: "⟦" + m.group(1) + "⟧", h, flags=re.S)
    h = re.sub(r'<svg\b([^>]*)>.*?</svg>',
               lambda m: "<svg" + m.group(1)[:300] + ">[svg body omitted]</svg>", h, flags=re.S)
    return h


for i, k in enumerate(keys[:-1]):
    blk = s[pos[k]:pos[keys[i + 1]]]
    open(os.path.join(OUT, k + ".html"), "w", encoding="utf-8").write(collapse(blk))
    p = P()
    p.feed(blk)
    p.flush()
    with open(os.path.join(OUT, k + ".txt"), "w", encoding="utf-8") as f:
        for sec, near, t in p.lines:
            f.write(f"[{near}] {t}\n")
    print(k, len(blk), len(p.lines))

assert hashlib.md5(open(PAGE, encoding="utf-8").read().encode("utf-8")).hexdigest() == md5_before
print("page unchanged", md5_before)

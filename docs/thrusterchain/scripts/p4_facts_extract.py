"""Extract reader-visible text from thrusterchain.html (read-only).

Writes notes/page_text.txt: one line per block element, prefixed with
[section-id | nearest-id] so each sentence can be located on the page.
MathJax containers are replaced by their aria-label between ⟦math: ...⟧.
Script and style contents are skipped (JS strings are dumped separately to
notes/js_strings.txt, excluding the large data payload line).
"""
import html
import re
from html.parser import HTMLParser

SRC = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"
OUT = "/mnt/c/Users/David/dev/bca-shift-work/page/tmp/p4/facts/page_text.txt"
OUT_JS = "/mnt/c/Users/David/dev/bca-shift-work/page/tmp/p4/facts/js_strings.txt"

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


def main():
    src = open(SRC, encoding="utf-8").read()
    p = P()
    p.feed(src)
    p.flush()
    with open(OUT, "w", encoding="utf-8") as f:
        for sec, near, t in p.lines:
            f.write(f"[{sec} | {near}] {t}\n")
    # JS string literals (skip giant lines)
    js_out = []
    for m in re.finditer(r"<script[^>]*>(.*?)</script>", src, re.S):
        body = m.group(1)
        start_line = src[: m.start()].count("\n") + 1
        for k, line in enumerate(body.split("\n")):
            if len(line) > 20000:
                js_out.append(f"L{start_line + k}: <skipped {len(line)} chars of data>")
                continue
            for s in re.findall(r"'((?:[^'\\]|\\.){3,})'|\"((?:[^\"\\]|\\.){3,})\"|`((?:[^`\\]|\\.){3,})`", line):
                lit = next(x for x in s if x)
                if re.search(r"[A-Za-z]{3,}.*\s", lit):
                    js_out.append(f"L{start_line + k}: {lit}")
    with open(OUT_JS, "w", encoding="utf-8") as f:
        f.write("\n".join(js_out) + "\n")
    print(len(p.lines), "text lines;", len(js_out), "js strings")


main()

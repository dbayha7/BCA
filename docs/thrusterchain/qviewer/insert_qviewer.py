#!/usr/bin/env python3
"""Insert the section "The critic in action" into a copy of the ThrusterChain page fragment.

Five insertions, each anchored on something the page has exactly once (the script refuses, and writes nothing, if an
anchor is missing, repeated, or the section is already there):
  1. style.css      just before the first </style> that comes before <div class="wrap">
  2. section.html   just before the line that holds <section id="methods">
  3. the contents   <a href="#critic-in-action">The critic in action</a>, just before the line of the link to
     rail link      #methods inside <nav class="toc">
  4. qv_data.json   as one <script type="application/json" id="qv-data"> at the end of the page
  5. script.js      as one <script> after it, at the very end
Each insertion is wrapped in begin/end markers (qviewer:begin / qviewer:end) so that --replace can take out an earlier
insertion first and put the current files in. Taking the blocks out gives back the original page byte for byte.
section.html goes in verbatim: its tables already hold their rows (a dash in each cell), so that the page's own table
tooltips prepare them when the page loads; script.js then only changes the cells' text.

Usage:
  python insert_qviewer.py PAGE [--out OUT] [--replace] [--dir DIR]
PAGE is read; OUT (default: PAGE itself) is written. Writing in place to a file inside a folder named "page" is refused
unless --in-place-page is given, so the working copy is never changed by accident.
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
B, E = "qviewer:begin", "qviewer:end"


class Refused(Exception):
    pass


def one(pattern, text, what, flags=0):
    hits = [m for m in re.finditer(pattern, text, flags)]
    if len(hits) != 1:
        raise Refused(f"refused: expected exactly one {what}, found {len(hits)}")
    return hits[0]


def line_start(text, i):
    return text.rfind("\n", 0, i) + 1


def strip_previous(page):
    """Take out an earlier insertion (every block between the markers). Returns the page and how many were taken."""
    n = 0
    pats = [r"\n?/\* qviewer:begin [^*]*\*/.*?/\* qviewer:end [^*]*\*/\n?",
            r"[ \t]*<!-- qviewer:begin [^>]*-->.*?<!-- qviewer:end [^>]*-->\n{0,2}"]
    for p in pats:
        page, k = re.subn(p, lambda m: "\n" if m.group(0).startswith("\n") and m.group(0).endswith("\n") else "", page,
                          flags=re.S)
        n += k
    return page, n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("page")
    ap.add_argument("--out")
    ap.add_argument("--dir", default=HERE, help="folder holding section.html, style.css, script.js, qv_data.json")
    ap.add_argument("--replace", action="store_true", help="take out an earlier insertion first")
    ap.add_argument("--in-place-page", action="store_true", help="allow writing in place inside a folder named page")
    a = ap.parse_args()
    out = a.out or a.page
    if os.path.abspath(out) == os.path.abspath(a.page) and \
            os.path.basename(os.path.dirname(os.path.abspath(a.page))) == "page" and not a.in_place_page:
        raise Refused("refused: this would change the page's working copy in place; pass --out, or --in-place-page")

    with open(a.page, encoding="utf-8", newline="") as fh:
        page = fh.read()
    rd = lambda n: open(os.path.join(a.dir, n), encoding="utf-8").read()
    css, sec, js = rd("style.css").rstrip() + "\n", rd("section.html").rstrip("\n") + "\n", rd("script.js").rstrip() + "\n"
    data_text = rd("qv_data.json").strip()

    # the files themselves
    data = json.loads(data_text)
    if data.get("schema") != "bca-shift-qviewer-page-v2":
        raise Refused(f"refused: qv_data.json has schema {data.get('schema')!r}")
    one(r'<section\s+id="critic-in-action"[\s>]', sec, 'section id="critic-in-action" in section.html')
    if "</script" in js.lower() or "<!--" in js:
        raise Refused("refused: script.js contains </script or <!--")
    if "</style" in css.lower():
        raise Refused("refused: style.css contains </style")
    data_safe = data_text.replace("</", "<\\/").replace("<!--", "<\\!--")
    assert json.loads(data_safe) == data

    if a.replace:
        page, n = strip_previous(page)
        print(f"took out {n} earlier inserted block(s)")
    for pat, what in ((r'id="critic-in-action"', 'id="critic-in-action"'), (r'id="qv-data"', 'id="qv-data"'),
                      (r'href="#critic-in-action"', 'a link to #critic-in-action'), (re.escape(B), "a qviewer marker")):
        if re.search(pat, page):
            raise Refused(f"refused: the page already has {what}; use --replace to put in the current files")

    # anchors (all found before anything changes)
    wrap = one(r'<div\s+class="wrap"\s*>', page, '<div class="wrap">')
    styles = [m for m in re.finditer(r"</style\s*>", page[:wrap.start()])]
    if not styles:
        raise Refused('refused: no </style> before <div class="wrap">')
    st = styles[0]
    meth = one(r'<section\s+id="methods"\s*>', page, '<section id="methods">')
    nav = one(r'<nav\s+class="toc"[^>]*>.*?</nav>', page, '<nav class="toc"> ... </nav>', re.S)
    link = one(r'<a\s+href="#methods"[^>]*>', nav.group(0), 'link to #methods inside the contents rail')
    link_i = nav.start() + link.start()
    if not (wrap.start() < meth.start() < nav.start()):
        raise Refused("refused: the page's order is not wrap, then the methods section, then the contents rail")
    if page.rstrip()[-9:].lower() != "</script>" and not page.rstrip().endswith(">"):
        raise Refused("refused: the page does not end with a tag")

    # build: insert from the end of the page backwards, so earlier offsets stay valid. The data and the script are
    # appended after the page's last character, which stays as it was, so taking the blocks out restores it exactly
    tail = (f"<!-- {B} data -->\n<script type=\"application/json\" id=\"qv-data\">{data_safe}</script>\n"
            f"<!-- {E} data -->\n<!-- {B} script -->\n<script>\n{js}</script>\n<!-- {E} script -->\n")
    new = page + tail
    ls = line_start(new, link_i)
    indent = re.match(r"[ \t]*", new[ls:]).group(0)
    new = (new[:ls] + f"{indent}<!-- {B} link --><a href=\"#critic-in-action\">The critic in action</a><!-- {E} link -->\n"
           + new[ls:])
    ms = line_start(new, meth.start())
    new = new[:ms] + f"  <!-- {B} section -->\n" + sec + f"  <!-- {E} section -->\n\n" + new[ms:]
    new = new[:st.start()] + f"/* {B} style */\n" + css + f"/* {E} style */\n" + new[st.start():]

    # checks on the result
    for pat, what in ((r'<section\s+id="critic-in-action"', "the section"),
                      (r'<script\s+type="application/json"\s+id="qv-data"\s*>', "the data tag"),
                      (r'href="#critic-in-action"', "the contents link"), (r'<section\s+id="methods"', "the methods section")):
        one(pat, new, what + " in the result")
    i_sec, i_meth = new.index('id="critic-in-action"'), re.search(r'<section\s+id="methods"', new).start()
    if not i_sec < i_meth:
        raise Refused("refused: the section did not land before the methods section")
    nav2 = one(r'<nav\s+class="toc"[^>]*>.*?</nav>', new, "contents rail in the result", re.S).group(0)
    if not nav2.index('href="#critic-in-action"') < nav2.index('href="#methods"'):
        raise Refused("refused: the contents link did not land before the methods link")
    w2 = re.search(r'<div\s+class="wrap"\s*>', new).start()
    if not new.index(f"/* {E} style */") < w2:
        raise Refused("refused: the style did not land before the wrap")
    with open(out, "w", encoding="utf-8", newline="") as fh:
        fh.write(new)
    print(f"wrote {out}: {len(page.encode('utf-8')):,} -> {len(new.encode('utf-8')):,} bytes "
          f"(data {len(data_safe.encode('utf-8')):,}, script {len(js.encode('utf-8')):,}, style {len(css.encode('utf-8')):,}, "
          f"section {len(sec.encode('utf-8')):,})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Refused as e:
        print(e, file=sys.stderr)
        sys.exit(2)

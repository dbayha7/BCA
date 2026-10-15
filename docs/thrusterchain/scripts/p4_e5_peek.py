"""Read-only: dump raw HTML of page parts with math containers and SVG bodies collapsed (edit 5).
usage: python3 scripts/p4_e5_peek.py [page] -- id1 id2 ...  -> tmp/p4/edit5/cur/<id>.html
an id may be a section id, or 'top' (wrap start to the first section)."""
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
PD = os.path.dirname(HERE)
args = sys.argv[1:]
page = os.path.join(PD, "thrusterchain.html")
if "--" in args:
    i = args.index("--")
    if i:
        page = args[0]
    args = args[i + 1:]
s = open(page, encoding="utf-8").read()
OUT = os.path.join(PD, "tmp", "p4", "edit5", "cur")
os.makedirs(OUT, exist_ok=True)
CONT = re.compile(r'<mjx-container role="img" aria-label="(.*?)" class="MathJax" jax="SVG"( display="true")?>.*?</mjx-container>', re.S)
for sid in args:
    if sid == "top":
        a = s.index('<div class="wrap"')
        b = s.index('<section id=', a)
    elif sid == "style":
        a = s.index('<style'); b = s.index('<div class="wrap"')
    elif sid == "tail":
        a = s.index('</section>', s.index('<section id="code-names"')); b = len(s)
    else:
        a = s.index(f'<section id="{sid}"')
        b = s.index("</section>", a) + len("</section>")
    t = s[a:b]
    t = CONT.sub(lambda m: f'[M{"D" if m.group(2) else ""}: {m.group(1)}]', t)
    t = re.sub(r'(<svg\b[^>]*>).*?(</svg>)', lambda m: m.group(1) + "…" + m.group(2), t, flags=re.S)
    open(os.path.join(OUT, sid + ".html"), "w", encoding="utf-8").write(t)
    print(sid, a, b - a, len(t), t.count("\n"))

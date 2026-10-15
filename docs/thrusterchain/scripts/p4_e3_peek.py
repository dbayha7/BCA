"""Read-only: dump raw HTML of sections with math containers and big SVGs collapsed, for reading.
usage: python3 scripts/p4_e3_peek.py [page] -- id1 id2 ...   -> tmp/p4/edit3/cur/<id>.html"""
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
PD = os.path.dirname(HERE)
args = sys.argv[1:]
page = os.path.join(PD, "thrusterchain.html")
if "--" in args:
    i = args.index("--"); 
    if i: page = args[0]
    args = args[i + 1:]
s = open(page, encoding="utf-8").read()
OUT = os.path.join(PD, "tmp", "p4", "edit3", "cur")
os.makedirs(OUT, exist_ok=True)
CONT = re.compile(r'<mjx-container role="img" aria-label="(.*?)" class="MathJax" jax="SVG"( display="true")?>.*?</mjx-container>', re.S)
for sid in args:
    a = s.index(f'<section id="{sid}"')
    b = s.index("</section>", a) + len("</section>")
    t = s[a:b]
    t = CONT.sub(lambda m: f'[M{"D" if m.group(2) else ""}: {m.group(1)}]', t)
    t = re.sub(r'(<svg\b[^>]*>).*?(</svg>)', lambda m: m.group(1) + "…" + m.group(2), t, flags=re.S)
    open(os.path.join(OUT, sid + ".html"), "w", encoding="utf-8").write(t)
    print(sid, b - a, len(t))

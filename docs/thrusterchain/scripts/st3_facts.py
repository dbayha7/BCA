"""Read-only: where each Version History fact also appears on the page (outside #version-history)."""
import re, sys
sys.path.insert(0, "/mnt/c/Users/David/dev/bca-shift-work/page/scripts")
P = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"
src = open(P, encoding="utf-8").read()
cut = src.find("\n<script>")
body = src[:cut]
vh0 = body.find('<section id="version-history">'); vh1 = body.find("</section>", vh0) + 10
pat = re.compile(r"\n(  <header class=\"head\" id=\"top\">|  <section id=\"([^\"]+)\">|    <figure class=\"panel\" id=\"sim\")")
m0 = body.find("<main>"); m1 = body.find("</main>")
starts = [(m.start() + 1, m.group(2) or ("top" if "header" in m.group(1) else "sim")) for m in pat.finditer(body, m0, m1)]
def where(i):
    if i < m0: return "head"
    if i > m1: return "after-main"
    b = "?"
    for s, bid in starts:
        if s <= i: b = bid
    return b
def strip(s):
    s = re.sub(r"<mjx-container[^>]*?aria-label=\"([^\"]*)\".*?</mjx-container>", lambda m: "[M:" + m.group(1) + "]", s, flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s))
terms = sys.argv[1:]
for t in terms:
    rx = re.compile(t, re.I)
    hits = [m for m in rx.finditer(body) if not (vh0 <= m.start() < vh1)]
    print(f"##### {t}: {len(hits)} hits outside Version History")
    for m in hits[:12]:
        ctx = strip(body[max(0, m.start() - 400): m.end() + 300])
        print(f"  - in {where(m.start())} @line {body.count(chr(10), 0, m.start()) + 1}: ...{ctx[-500:] if len(ctx) > 500 else ctx}...")

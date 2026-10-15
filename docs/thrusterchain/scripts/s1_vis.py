"""Read-only: visible text of a top-level block (formulas as [M:tex]); optional --tips also prints data-tip texts.
Usage: s1_vis.py BLOCK [--tips]"""
import re, sys, html as H
P = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"
s = open(P, encoding="utf-8").read()
m0 = s.find("<main>"); m1 = s.find("</main>")
pat = re.compile(r"\n(  <header class=\"head\" id=\"top\">|  <section id=\"([^\"]+)\">|    <figure class=\"panel\" id=\"sim\")")
st = [(m.start() + 1, m.group(2) or ("top" if "header" in m.group(1) else "sim")) for m in pat.finditer(s, m0, m1)]
blocks = {}
for i, (a, bid) in enumerate(st):
    blocks[bid] = s[a: st[i + 1][0] if i + 1 < len(st) else m1]
def vis(x):
    x = re.sub(r"<mjx-container[^>]*?aria-label=\"([^\"]*)\".*?</mjx-container>", lambda m: "[M:" + m.group(1) + "]", x, flags=re.S)
    x = re.sub(r"<svg\b.*?</svg>", " [svg] ", x, flags=re.S)
    x = re.sub(r"<script\b.*?</script>", " ", x, flags=re.S)
    x = re.sub(r"\sdata-tip[a-z-]*=\"[^\"]*\"", "", x)
    x = re.sub(r"</(p|li|h2|h3|tr|div|dt|dd)>", "\n", x)
    return re.sub(r"[ \t]+", " ", H.unescape(re.sub(r"<[^>]+>", "", x)))
b = blocks[sys.argv[1]]
print(re.sub(r"\n\s*\n+", "\n", vis(b)))
if "--tips" in sys.argv:
    for t, d in re.findall(r'data-tip-title="([^"]*)" data-tip="([^"]*)"', b):
        print("TIP", H.unescape(t), "::", H.unescape(d)[:200])

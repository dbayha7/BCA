"""CPU 2: numbers in the draft's After-the-stop prose that do not occur in the page's After-the-stop subsection (read only)."""
import html, re
PAGE = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"
DRAFT = "/mnt/c/Users/David/dev/bca-shift-work/page/notes/results_draft.html"
NUM = re.compile(r"(?<![\w.])[−+-]?\d[\d,]*(?:\.\d+)?")
def toks(t):
    return [m.group(0).replace("−", "-").replace("+", "").replace(",", "") for m in NUM.finditer(t)]
s = open(PAGE, encoding="utf-8").read()
i = s.index('id="beyond-the-data"'); j = s.index('id="weighting-test"')
pg = html.unescape(re.sub(r"<[^>]+>", " ", s[i:j]))
pg += " ".join(html.unescape(a) for a in re.findall(r'(?:data-tip|aria-label)="([^"]*)"', s[i:j]))
pt = set(toks(pg)); pt |= {x.lstrip("-") for x in pt}
d = open(DRAFT, encoding="utf-8").read()
d = re.sub(r"<!--.*?-->", " ", d, flags=re.S); d = d[d.index(chr(105)+chr(100)+"=\"beyond-the-data\""):]
d = re.sub(r"<!--.*?-->", " ", d, flags=re.S); d = re.sub(r"\[MATH(?:-DISPLAY)?:[^\]]*\]", " ", d)
dt = html.unescape(re.sub(r"<[^>]+>", " ", d))
for m in NUM.finditer(dt):
    t = m.group(0).replace("−", "-").replace("+", "").replace(",", "")
    if t not in pt and t.lstrip("-") not in pt:
        print(repr(m.group(0)), "...", re.sub(r"\s+", " ", dt[max(0, m.start()-60): m.start()+30]))
print("done")

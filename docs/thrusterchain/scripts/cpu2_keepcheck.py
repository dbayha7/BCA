"""CPU 2: numbers in the page's Results that the draft's own text does not repeat, by page block (read only).
Those blocks must be among the draft's KEEP blocks."""
import html, re
PAGE = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"
DRAFT = "/mnt/c/Users/David/dev/bca-shift-work/page/notes/results_draft.html"
NUM = re.compile(r"(?<![\w.])[−+-]?\d[\d,]*(?:\.\d+)?")
def norm(t): return t.replace("−", "-").replace("+", "").replace(",", "")
s = open(PAGE, encoding="utf-8").read()
i = s.index('<section id="results"'); j = s.index('<section id="weighting-test"')
sec = s[i:j]
sec = re.sub(r"<mjx-container.*?</mjx-container>", " ", sec, flags=re.S)
sec = re.sub(r"<svg\b.*?</svg>", " ", sec, flags=re.S)
d = open(DRAFT, encoding="utf-8").read()
d = re.sub(r"<!--.*?-->", " ", d, flags=re.S)
dt = {norm(m.group(0)) for m in NUM.finditer(html.unescape(re.sub(r"<[^>]+>", " ", d)))}
dt |= {x.lstrip("-") for x in dt}
# split the page section at headings so each missing number can be placed
parts = re.split(r"(<h[34][^>]*>.*?</h[34]>)", sec, flags=re.S)
head = "Results top"
for p in parts:
    if re.match(r"<h[34]", p):
        head = re.sub(r"<[^>]+>", "", p).strip()[:70]; continue
    txt = html.unescape(re.sub(r"<[^>]+>", " ", p))
    miss = sorted({norm(m.group(0)) for m in NUM.finditer(txt)} - dt, key=lambda x: (len(x), x))
    if miss:
        print(f"[{head}] {len(miss)}: {' '.join(miss[:40])}{' ...' if len(miss) > 40 else ''}")

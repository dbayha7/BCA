"""CPU 2: check every number in the two drafts' visible text against the current page and the v5 reports (read only).

A number passes if its token appears among the number tokens of the current page (text, data-tip, aria-label), or among
values formatted from the v5 reports and SPEC anchors. Everything else is printed for review (expected: the derived
numbers listed in the drafts' comments, and plain counts or ranks such as 1, 2, 3).
"""
import html
import json
import re

PAGE = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"
DRAFTS = ["/mnt/c/Users/David/dev/bca-shift-work/page/notes/results_draft.html",
          "/mnt/c/Users/David/dev/bca-shift-work/page/notes/weighting_results_draft.html"]
QS = ["random", "medium-replay", "medium", "medium-expert", "expert"]
ROOT = "/root/bca-shift/runs/shift_iw_v5"
NUM = re.compile(r"(?<![\w.])[−-]?\d[\d,]*(?:\.\d+)?")


def norm(t):
    t = t.replace("−", "-").replace(",", "")
    if t.startswith("+"):
        t = t[1:]
    return t


def tokens(text):
    return {norm(m.group(0)) for m in NUM.finditer(text)}


page = open(PAGE, encoding="utf-8").read()
page_txt = html.unescape(re.sub(r"<[^>]+>", " ", page))
attrs = " ".join(html.unescape(a) for a in re.findall(r'(?:data-tip|aria-label|data-tip-title)="([^"]*)"', page))
page_tok = tokens(page_txt + " " + attrs)
# also allow the absolute value of page numbers (prose may say "falls by 3.9" for "−3.9")
page_tok |= {t.lstrip("-") for t in page_tok}

v5 = set()


def add(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return
    for s in (1, 100):
        for d in range(0, 4):
            v = f"{x * s:.{d}f}"
            v5.add(v)
            v5.add(v.lstrip("-"))


def walk(o):
    if isinstance(o, dict):
        for v in o.values():
            walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)
    else:
        add(o)


for q in QS:
    walk(json.load(open(f"{ROOT}/{q}/report/report.json")))
walk(json.load(open(f"{ROOT}/summary/summary.json")))
for v in [-800.29, -35.19, -1061.42, -54.66, -1351.41, -80.61, 990, 1140, 16.5, 19, 1.5, 20, 24, 1.8, 89, 94, 96, 88,
          86, 91.7, 92, 82, 87, 23, 28, 33, 39, 99, 500, 300, 42500, 843, 1000, 100000, 65, 5, 0.053, 0.039]:
    add(v)

for path in DRAFTS:
    s = open(path, encoding="utf-8").read()
    vis = re.sub(r"<!--.*?-->", " ", s, flags=re.S)
    vis = re.sub(r"\[MATH(?:-DISPLAY)?:[^\]]*\]", " ", vis)
    vis = html.unescape(re.sub(r"<[^>]+>", " ", vis))
    found = [(m.group(0), m.start()) for m in NUM.finditer(vis)]
    miss = []
    for tok, pos in found:
        t = norm(tok)
        if t in page_tok or t.lstrip("-") in page_tok or t in v5:
            continue
        ctx = vis[max(0, pos - 50): pos + 30].replace("\n", " ")
        miss.append((tok, re.sub(r"\s+", " ", ctx)))
    print(f"== {path.rsplit('/', 1)[1]}: {len(found)} numbers, {len(miss)} not found on the page or in the v5 files")
    for tok, ctx in miss:
        print(f"   {tok:>10s}  ...{ctx}...")

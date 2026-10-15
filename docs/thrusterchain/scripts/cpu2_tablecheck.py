"""CPU 2: compare the numeric cells of the rewritten Results tables with the current page's tables (read only)."""
import html
import re

PAGE = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"
DRAFT = "/mnt/c/Users/David/dev/bca-shift-work/page/notes/results_draft.html"


def section(s, sid):
    i = s.index(f'<section id="{sid}"')
    depth = 0
    for t in re.finditer(r"<(/?)section\b", s[i:]):
        depth += -1 if t.group(1) else 1
        if depth == 0:
            return s[i:i + t.end()]


def tables(s):
    out = []
    for t in re.findall(r"<table\b.*?</table>", s, flags=re.S):
        body = re.search(r"<tbody>(.*?)</tbody>", t, flags=re.S)
        if not body:
            continue
        cells = re.findall(r"<td\b[^>]*>(.*?)</td>", body.group(1), flags=re.S)
        txt = [re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", c))).strip() for c in cells]
        nums = [n for c in txt for n in re.findall(r"[−+-]?\d[\d,]*(?:\.\d+)?", c)]
        out.append((txt[0][:50] if txt else "", nums))
    return out


page = section(open(PAGE, encoding="utf-8").read(), "results")
page = page[:page.index('id="beyond-the-data"')]
draft = open(DRAFT, encoding="utf-8").read()
draft = re.sub(r"<!--.*?-->", "", draft, flags=re.S)
draft = draft[:draft.index('id="beyond-the-data"')]
pt, dt = tables(page), tables(draft)
print("page tables:", [p[0] for p in pt])
print("draft tables:", [d[0] for d in dt])
pairs = {"critic": (0, 0), "cost": (1, 1), "variants": (2, 3), "rule": (3, 4), "world": (4, 5)}
for name, (i, j) in pairs.items():
    a, b = pt[i][1], dt[j][1]
    a2 = [x.replace("+", "") for x in a]
    b2 = [x.replace("+", "") for x in b]
    # the draft's first-cell text adds words with numbers (e.g. "1/1.5"); compare as multisets of the page's numbers
    missing = [x for x in a2 if x not in b2]
    extra = [x for x in b2 if x not in a2]
    print(f"{name}: page {len(a2)} numbers, draft {len(b2)}; page numbers missing from draft: {missing}; draft numbers not in page table: {extra}")
    if a2 == b2:
        print("   identical sequence")

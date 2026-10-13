"""Edit 4 checks on a built page (default tmp/r4/applied.html); read-only.
  wsl.exe -d Ubuntu-22.04 -u root -e bash -lc 'python3 /mnt/c/Users/David/dev/bca-shift-work/page/scripts/r4_verify.py [page]'
1. Results and the weighting test nest correctly (an HTML parser walks them; math SVG is skipped).
2. Every in-page link resolves; the contents rail lists ids that exist.
3. Math: every glyph id unique, every <use> inside its own SVG.
4. Wording: no informal words, nicknames or starred symbols in the two sections' visible text and tooltips.
5. Numbers: every number in the two sections' visible text and tooltips is already on the page before this edit, or is a
   version-5 number (rounded from the report and summary files), or is in the short list of derived numbers below.
6. Tables: each table of the two sections gets a column-description entry for every column (the script's own rule).
"""
import html as H
import json
import re
import sys
from html.parser import HTMLParser

sys.path.insert(0, "/mnt/c/Users/David/dev/bca-shift-work/page/scripts")
import mjx  # noqa: E402

P = "/mnt/c/Users/David/dev/bca-shift-work/page/"
PAGE = sys.argv[1] if len(sys.argv) > 1 else P + "tmp/r4/applied.html"
PRE = P + "tmp/r4/thrusterchain.pre-edit4.html"
V5 = "/root/bca-shift/runs/shift_iw_v5"
QS = ["random", "medium-replay", "medium", "medium-expert", "expert"]
DERIVED = {"1270.8": "−80.61 − (−1351.41)", "6.0": "W1 strong − W2, medium-replay (91.00 − 84.97)",
           "11.1": "W1 strong − W2, medium-expert (94.62 − 83.58)", "51": "tables on the page", "2020": "year of a cited paper"}
s = open(PAGE, encoding="utf-8").read()
pre = open(PRE, encoding="utf-8").read()
fails = []


def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what)
    if not ok:
        fails.append(what)


def section(t, sid):
    a = t.index('<section id="%s"' % sid)
    return t[a:t.index("</section>", a) + len("</section>")]


RES, WT = section(s, "results"), section(s, "weighting-test")
nomath = lambda t: re.sub(r"<mjx-container.*?</mjx-container>", " ", t, flags=re.S)

# 1. nesting
VOID = {"br", "img", "input", "hr", "meta", "link", "wbr", "col", "source"}


class Nest(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack, self.err = [], []

    def handle_starttag(self, tag, attrs):
        if tag not in VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.err.append(f"</{tag}> closes <{self.stack[-1] if self.stack else None}>")
            if tag in self.stack:
                while self.stack and self.stack.pop() != tag:
                    pass
        else:
            self.stack.pop()


for name, t in (("Results", RES), ("weighting test", WT)):
    n = Nest()
    n.feed(nomath(t))
    check(not n.err and not n.stack, f"{name}: every tag closes in order ({n.err[:3]}, open {n.stack[:5]})")

# 2. links
ids = set(re.findall(r'\sid="([^"]+)"', s))
hrefs = re.findall(r'href="#([^"]+)"', nomath(s))
bad = sorted({h for h in hrefs if h not in ids})
check(not bad, f"all {len(hrefs)} in-page links resolve (missing: {bad})")
rail = re.findall(r'<a href="#([^"]+)"', s[s.index('<nav class="toc"'):s.index("</nav>", s.index('<nav class="toc"'))])
check(all(r in ids for r in rail), f"the rail's {len(rail)} links all exist")

# 3. math
check(mjx.check(s) > 0, f"math: {mjx.check(s)} glyph ids unique, every <use> resolves")

# 4. wording
def text_and_tips(t):
    t = nomath(t)
    tips = " ".join(H.unescape(x) for x in re.findall(r'data-tip(?:-title)?="([^"]*)"', t))
    vis = H.unescape(re.sub(r"<[^>]+>", " ", re.sub(r"<script.*?</script>", " ", t, flags=re.S)))
    return re.sub(r"\s+", " ", vis + " " + tips)


TXT = text_and_tips(RES) + " " + text_and_tips(WT)
BANNED = [r"\bfavou?rite", r"\bhill\b", r"\bnudge", r"\bwobble", r"\btrusts?\b", r"\bknows?\b", r"\btoy\b", r"\bgate\b",
          r"gate's", r"\bgapped\b", r"room check", r"power check", r"\bthe user\b", r"caution can'?t help",
          r"caution cannot help", r"\ba\*", r"δ\*", r"\brecorder\b", r"\broom\b", r"\breach\)", r"\breach and\b", r"illustration"]
hits = {b: re.findall(r".{0,40}" + b + r".{0,40}", TXT, flags=re.I) for b in BANNED}
hits = {b: h for b, h in hits.items() if h}
check(not hits, f"no informal words, nicknames or stars in the two sections ({hits})")
hdr = s[s.index('<header class="head"'):s.index("</header>")]
check("cannot win back that lost return, in any situation" not in hdr and "could not win back the return lost in strong"
      in hdr, "header: the narrower claim replaces the broad one")

# 5. numbers
NUM = re.compile(r"(?<![\w.])[−+-]?\d[\d,]*(?:\.\d+)?")


def nums(t):
    out = []
    for m in NUM.finditer(t):
        x = m.group(0).replace("−", "-").replace(",", "").lstrip("+")
        try:
            out.append(abs(float(x)))
        except ValueError:
            pass
    return out


allowed = set(nums(text_and_tips(pre)))
allowed |= set(nums(re.sub(r"<[^>]+>", " ", pre)))


def walk(x):
    if isinstance(x, bool):
        return
    if isinstance(x, (int, float)):
        v = abs(float(x))
        for d in range(0, 4):
            allowed.add(round(v, d))
            allowed.add(round(100 * v, d))
        return
    if isinstance(x, dict):
        for v in x.values():
            walk(v)
    if isinstance(x, list):
        for v in x:
            walk(v)


try:
    for q in QS:
        walk(json.load(open(f"{V5}/{q}/report/report.json")))
    walk(json.load(open(f"{V5}/summary/summary.json")))
    v5 = True
except OSError:
    v5 = False
spec = open("/mnt/c/Users/David/dev/bca-shift-work/iw/SPEC.md", encoding="utf-8").read() + \
    open("/mnt/c/Users/David/dev/bca-shift-work/iw/MARKS_FOR_USER.md", encoding="utf-8").read()
allowed |= set(nums(spec))
allowed |= {abs(float(k)) for k in DERIVED}
new = nums(TXT)
unknown = sorted({x for x in new if x not in allowed})
check(v5, "the version-5 report and summary files were read")
check(not unknown, f"every number ({len(new)}) is on the page before, in the version-5 files or derived; unknown: {unknown}")

# 6. tables and their column descriptions
cols = json.loads(re.search(r"  const COLS = (.*?);\n", s).group(1).replace("<\\/", "</"))


def names(table):
    th = re.findall(r"<thead>.*?</thead>", table, flags=re.S)
    last = re.findall(r"<tr>(.*?)</tr>", th[0], flags=re.S)[-1] if th else ""
    return [re.sub(r"\s+", " ", H.unescape(re.sub(r"<[^>]+>", "", c))).strip() for c in re.findall(r"<th[^>]*>(.*?)</th>", last, flags=re.S)]


for sid, t in (("results", RES), ("weighting-test", WT)):
    used = set()
    for tb in re.findall(r"<table.*?</table>", nomath(t), flags=re.S):
        nm = names(tb)
        hit = None
        for i, e in enumerate(cols[sid]):
            if i in used:
                continue
            if all(n in e["cols"] for n in nm):
                hit = i
                break
        check(hit is not None, f"{sid}: table {nm[:2]}… has a column-description entry")
        if hit is not None:
            used.add(hit)
            for n in nm:
                d = e["cols"][n]
                d = d[1] if isinstance(d, list) else d
                check(d.endswith(".") and d[0].isupper(), f"{sid}: '{n}' description is a sentence")
    check(len(used) == len(cols[sid]), f"{sid}: every entry used ({len(used)} of {len(cols[sid])})")
ntab = len(re.findall(r"<table[\s>]", nomath(re.sub(r"<script.*?</script>", "", s, flags=re.S))))
check(ntab == 51, f"{ntab} tables on the page (48 before; +1 in Results, +2 in the weighting test)")

print(f"\n{len(fails)} failure(s)")
sys.exit(1 if fails else 0)

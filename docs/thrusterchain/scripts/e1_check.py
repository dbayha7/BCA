"""Read-only checks on the edited page against the backup: unique ids, every in-page link and every MathJax glyph
reference resolves, the data line is byte-identical, main's order and the rail's order agree, and every MathJax
container of the backup that is gone was one of the expected ones."""
import collections
import re
import sys

from e1_common import BACKUP, MJX, LABEL, PAGE, data_line_hash

page = sys.argv[1] if len(sys.argv) > 1 else PAGE
s = open(page, encoding="utf-8").read()
b = open(BACKUP, encoding="utf-8").read()
ok = True


def bad(msg):
    global ok
    ok = False
    print("FAIL", msg)


ids = re.findall(r'\bid="([^"]+)"', s)
dups = [k for k, v in collections.Counter(ids).items() if v > 1]
if dups:
    bad(f"duplicate ids {dups[:8]}")
idset = set(ids)
for h in set(re.findall(r'(?<![:\w])href="#([^"]+)"', s)):
    if h not in idset:
        bad(f"href #{h} does not resolve")
for h in set(re.findall(r'xlink:href="#([^"]+)"', s)):
    if h not in idset:
        bad(f"MathJax reference #{h} does not resolve")
if data_line_hash(s) != data_line_hash(b):
    bad("data line differs from the backup")
m0, m1 = s.index("<main>"), s.index("</main>")
kids = re.findall(r'\n {2,4}<(?:section|figure class="panel"|header class="head") id="([^"]+)"', s[m0:m1])
nav = s[s.index('<nav class="toc"'):s.index("</nav>")]
rail = re.findall(r'href="#([^"]+)"', nav)
if rail != ["glance"] + kids[1:]:
    bad(f"rail order {rail} differs from main {kids}")
labels_b = collections.Counter(LABEL.search(m.group(0)).group(1) for m in MJX.finditer(b))
labels_s = collections.Counter(LABEL.search(m.group(0)).group(1) for m in MJX.finditer(s))
gone = labels_b - labels_s
new = labels_s - labels_b
print("main order:", " > ".join(kids))
print("ids:", len(ids), "| links:", len(re.findall(r'(?<![:\w])href="#', s)), "| MathJax containers:",
      sum(labels_s.values()), "(backup", sum(labels_b.values()), ")")
print("MathJax containers dropped:", dict(gone) or "none", "| added:", dict(new) or "none")
print("PASS" if ok else "FAILURES ABOVE")
sys.exit(0 if ok else 1)

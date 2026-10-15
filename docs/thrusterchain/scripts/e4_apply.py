"""Edit 4: the reviewers' findings on the edited page.

usage (WSL): python3 e4_apply.py [--dry]
  Rebuilds the page from page/tmp/e4_base.html (the page as it stood after edit 3, md5 5334df16...). It refuses to
  write if the page was changed by anyone else since (its md5 must be the base's or this script's last output). The
  backup thrusterchain.v5.html is never touched.
How: every MathJax container and the data line are replaced by tokens (e3_common.tokenize), so no edit can change math
or data by accident. Each edit in e4_edits.py is (old, new): `old` must match exactly once, with any run of whitespace
in `old` matching any run of whitespace in the page. The math tokens listed in e4_edits.DROP may be removed (symbols
taken out of the live panel's readout labels); every other token must be put back exactly once.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from e3_common import BACKUP, PAGE, data_line_hash, md5  # noqa: E402
import e3_common  # noqa: E402
import e4_edits  # noqa: E402

BASE = "/mnt/c/Users/David/dev/bca-shift-work/page/tmp/e4_base.html"
STAMP = os.path.join(HERE, ".e4_last_md5")
dry = "--dry" in sys.argv

assert md5(BACKUP) == "0541be9517f03390d4884f84e667cc09", "the backup changed"
assert md5(BASE) == "5334df166d6527d512f61f1b9508163d", "the edit-4 base changed"
allowed = {md5(BASE)}
if os.path.exists(STAMP):
    allowed.add(open(STAMP).read().strip())
cur = md5(PAGE)
if cur not in allowed and not dry:
    sys.exit(f"refusing: {PAGE} was changed by someone else since edit 4 began (md5 {cur})")

src = open(BASE, encoding="utf-8").read()
t, store = e3_common.tokenize(src)
n_math = len(store) - 1

failures = []
for k, (old, new) in enumerate(e4_edits.EDITS):
    words = old.split()
    pat = r"\s+".join(re.escape(w) for w in words)
    m = list(re.finditer(pat, t))
    if len(m) != 1:
        failures.append(f"edit {k} ({len(m)} matches): {old[:140]!r}")
        continue
    a, b = m[0].start(), m[0].end()
    t = t[:a] + new.strip() + t[b:]   # a match never includes the whitespace around `old`, so none is added
if failures:
    print("\n".join(failures))
    sys.exit(f"{len(failures)} edits failed; nothing written")

# put the math and the data back; the DROP tokens may be gone
dropped = 0
for tok in list(store):
    if tok.startswith("⟦M") and tok.split("|")[0][2:] in e4_edits.DROP and t.count(tok) == 0:
        del store[tok]
        dropped += 1
out = e3_common.detokenize(t, store)
assert dropped == len(e4_edits.DROP), f"dropped {dropped} math tokens, expected {len(e4_edits.DROP)}"
assert data_line_hash(out) == data_line_hash(src), "data line changed"
assert out.count("<mjx-container") == src.count("<mjx-container") - dropped == n_math - dropped, "math count changed"
assert out.count("<script") == src.count("<script") and out.count("</script>") == src.count("</script>")
print(f"{len(e4_edits.EDITS)} edits applied; {n_math - dropped} math containers kept ({dropped} removed on purpose), "
      f"data line unchanged; {len(src)} -> {len(out)} chars")
if dry:
    open(os.path.join(os.path.dirname(PAGE), "tmp", "e4_dry.html"), "w", encoding="utf-8").write(out)
    print("dry run: wrote tmp/e4_dry.html")
else:
    open(PAGE, "w", encoding="utf-8").write(out)
    open(STAMP, "w").write(md5(PAGE))
    print("wrote", PAGE, md5(PAGE))

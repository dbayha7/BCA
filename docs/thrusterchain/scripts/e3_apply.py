"""Edit 3 (comment 1): the plain-language pass.

usage (WSL): python3 e3_apply.py [--dry]
  Rebuilds the page from page/tmp/e3_base.html (the page as it stood after edit 2). It refuses to write if the page was
  changed by anyone else since (its md5 must be the base's or this script's last output). The backup is never touched.
How: every MathJax container and the data line are replaced by tokens (e3_common.tokenize), so no edit can change math
or data. Each edit in e3_edits.py is (old, new): `old` must match exactly once, with any run of whitespace in `old`
matching any run of whitespace in the page. Then the tokens are put back, each exactly once.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from e3_common import BACKUP, BASE, PAGE, data_line_hash, detokenize, md5, tokenize  # noqa: E402
import e3_edits  # noqa: E402

STAMP = os.path.join(HERE, ".e3_last_md5")
dry = "--dry" in sys.argv

assert md5(BACKUP) == "0541be9517f03390d4884f84e667cc09", "the backup changed"
allowed = {md5(BASE)}
if os.path.exists(STAMP):
    allowed.add(open(STAMP).read().strip())
cur = md5(PAGE)
if cur not in allowed and not dry:
    sys.exit(f"refusing: {PAGE} was changed by someone else since edit 3 began (md5 {cur})")

src = open(BASE, encoding="utf-8").read()
t, store = tokenize(src)
n_math = len(store) - 1

failures = []
for k, (old, new) in enumerate(e3_edits.EDITS):
    words = old.split()
    pat = r"\s+".join(re.escape(w) for w in words)
    m = list(re.finditer(pat, t))
    if len(m) != 1:
        failures.append(f"edit {k} ({len(m)} matches): {old[:110]!r}")
        continue
    a, b = m[0].start(), m[0].end()
    t = t[:a] + new + t[b:]
if failures:
    print("\n".join(failures))
    sys.exit(f"{len(failures)} edits failed; nothing written")

out = detokenize(t, store)
assert data_line_hash(out) == data_line_hash(src), "data line changed"
assert out.count("<mjx-container") == src.count("<mjx-container") == n_math, "math count changed"
print(f"{len(e3_edits.EDITS)} edits applied; {n_math} math containers and the data line unchanged; "
      f"{len(src)} -> {len(out)} chars")
if dry:
    open(os.path.join(os.path.dirname(PAGE), "tmp", "e3_dry.html"), "w", encoding="utf-8").write(out)
    print("dry run: wrote tmp/e3_dry.html")
else:
    open(PAGE, "w", encoding="utf-8").write(out)
    open(STAMP, "w").write(md5(PAGE))
    print("wrote", PAGE, md5(PAGE))

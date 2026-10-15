"""Replace the column descriptions (the one-line COLS object) in the page's tooltip script with e5_cols.py's."""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from e5_cols import COLS  # noqa: E402

PAGE = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"
BACKUP = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.v9.html"
b0 = hashlib.md5(open(BACKUP, "rb").read()).hexdigest()
s = open(PAGE, encoding="utf-8").read()
head = "  const COLS = "
assert s.count(head) == 1, s.count(head)
i = s.index(head)
j = s.index(";\n", i)
old = s[i:j + 2]
json.loads(old[len(head):-2])                      # the old object is one JSON line
new = head + json.dumps(COLS, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/") + ";\n"
assert s.count(old) == 1
s = s.replace(old, new, 1)
open(PAGE, "w", encoding="utf-8").write(s)
assert hashlib.md5(open(BACKUP, "rb").read()).hexdigest() == b0
print("updated COLS:", len(old), "->", len(new), "chars; md5", hashlib.md5(s.encode("utf-8")).hexdigest())

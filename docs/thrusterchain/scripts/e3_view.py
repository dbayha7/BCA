"""Write a readable copy of a page (default: the edit-3 base) with math and data replaced by tokens.
usage: python3 e3_view.py [SRC] [OUT]"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e3_common import BASE, tokenize  # noqa: E402

src = sys.argv[1] if len(sys.argv) > 1 else BASE
out = sys.argv[2] if len(sys.argv) > 2 else "/mnt/c/Users/David/dev/bca-shift-work/page/tmp/e3_tok.html"
t, store = tokenize(open(src, encoding="utf-8").read())
open(out, "w", encoding="utf-8").write(t)
print(out, len(t), "chars,", len(store) - 1, "math tokens")

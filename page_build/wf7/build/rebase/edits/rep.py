"""Exact replacement helper for the rebase edit scripts: every old text must occur exactly as often as stated in its
file, or nothing is written (all files are checked before any is written)."""
import sys
from pathlib import Path

W7 = Path(__file__).resolve().parents[3]          # page_build/wf7


def apply(edits):
    """edits: list of (relative path under wf7, old, new, count)."""
    texts = {}
    for rel, old, new, count in edits:
        p = W7 / rel
        if rel not in texts:
            texts[rel] = p.read_text(encoding="utf-8")
        n = texts[rel].count(old)
        if n != count:
            sys.exit("%s: %r occurs %d times, expected %d; nothing written" % (rel, old[:100], n, count))
        texts[rel] = texts[rel].replace(old, new)
        if "\r" in new:
            sys.exit("CR byte in a replacement; nothing written")
    for rel, t in texts.items():
        (W7 / rel).write_text(t, encoding="utf-8", newline="\n")
        print("wrote", rel)

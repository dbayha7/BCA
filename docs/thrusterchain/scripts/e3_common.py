"""Shared helpers for edit 3 (the plain-language pass).

Tokenising: every MathJax container is replaced by a token that names its number and its aria-label, and the data line
`const D = ...;` by one token, so the rewrite works on readable text and can never change math or data. Detokenising
puts every original back, and asserts each token is used exactly once.
"""
import hashlib
import re

PAGE = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"
BACKUP = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.v5.html"
BASE = "/mnt/c/Users/David/dev/bca-shift-work/page/tmp/e3_base.html"

MJX = re.compile(r'<mjx-container\b[^>]*>.*?</mjx-container>', re.S)
LABEL = re.compile(r'aria-label="([^"]*)"')
DATA = re.compile(r'\nconst D = [^\n]*')


def tokenize(s):
    store = {}
    n = [0]

    def rep(m):
        n[0] += 1
        lab = LABEL.search(m.group(0))
        lab = lab.group(1) if lab else "?"
        tok = f"⟦M{n[0]}|{lab}⟧"
        store[tok] = m.group(0)
        return tok

    t = MJX.sub(rep, s)
    dm = list(DATA.finditer(t))
    assert len(dm) == 1, f"data line: expected one, found {len(dm)}"
    store["⟦DATA⟧"] = dm[0].group(0)[1:]
    t = t[:dm[0].start() + 1] + "⟦DATA⟧" + t[dm[0].end():]
    return t, store


def detokenize(t, store):
    for tok, orig in store.items():
        c = t.count(tok)
        assert c == 1, f"token {tok[:60]} used {c} times"
        t = t.replace(tok, orig)
    assert "⟦" not in t and "⟧" not in t, "a stray token bracket is left"
    return t


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def data_line_hash(s):
    m = list(DATA.finditer(s))
    assert len(m) == 1
    return hashlib.sha256(m[0].group(0).encode("utf-8")).hexdigest()

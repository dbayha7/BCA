"""Shared helpers for the edit-1 (structure) scripts. Every edit asserts that its anchor occurs exactly once."""
import hashlib
import re

PAGE = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"
BACKUP = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.v5.html"

MJX = re.compile(r'<mjx-container\b[^>]*>.*?</mjx-container>', re.S)
LABEL = re.compile(r'aria-label="([^"]*)"')


def once(hay, needle, what, start=0, end=None):
    end = len(hay) if end is None else end
    c = hay.count(needle, start, end)
    assert c == 1, f"{what}: expected exactly one match, found {c}"
    return hay.index(needle, start, end)


def replace_once(s, old, new, what, start=0, end=None):
    i = once(s, old, what, start, end)
    return s[:i] + new + s[i + len(old):]


def sub_ws(s, old_words, new, what, start=0, end=None):
    """Replace a passage given as words; any run of whitespace in the page matches any run in old_words."""
    end = len(s) if end is None else end
    pat = r"\s+".join(re.escape(w) for w in old_words.split())
    m = list(re.finditer(pat, s[start:end]))
    assert len(m) == 1, f"{what}: expected exactly one match, found {len(m)}"
    a, b = start + m[0].start(), start + m[0].end()
    return s[:a] + new + s[b:]


def section_span(s, sid):
    """(a, b): from the start of the line '  <section id="sid">' to just after its '  </section>\\n'."""
    marker = f'\n  <section id="{sid}">'
    a = once(s, marker, f"section {sid}") + 1
    end_marker = "\n  </section>\n"
    b = s.index(end_marker, a) + len(end_marker)
    assert s[a:b].count("<section") == 1, f"section {sid} is nested or malformed"
    return a, b


def cut(s, a, b):
    """Remove s[a:b] and one blank line after it; return (new_s, block)."""
    block = s[a:b]
    b2 = b + 1 if s[b:b + 1] == "\n" else b
    return s[:a] + s[b2:], block


def insert_after_section(s, sid, block):
    a, b = section_span(s, sid)
    return s[:b] + "\n" + block + s[b:]


def strip_text(html):
    """Visible text with tags removed, each MathJax container kept as its label, whitespace collapsed."""
    t = MJX.sub(lambda m: " [" + (LABEL.search(m.group(0)).group(1) if LABEL.search(m.group(0)) else "?") + "] ", html)
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def data_line_hash(s):
    i = once(s, "\nconst D = ", "data line") + 1
    j = s.index("\n", i)
    return hashlib.sha256(s[i:j].encode("utf-8")).hexdigest()


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

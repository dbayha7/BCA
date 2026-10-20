"""Near-80 (version 9, main setting): section 14 of PREREGISTRATION.md (the claim, word for word, for each outcome)
read into a machine-readable block, and the small primitives that fill its sentences in. DRAFT, not frozen.

extract_section14(md_text) returns the block that prereg.json carries as 'claims_word_for_word' (the freeze copies it
verbatim; the looks read their sentences from it):
  source, intro, brackets (every row of the bracket table, in order), always_attached, smoke_prefix,
  sentences ({id: {section, label, template text}}), fill_rules (the bullets of section 14.5, verbatim).
A template is the quoted sentence with the blockquote markers removed, its lines joined by one space and its outer
double quotes removed. The extraction refuses (ValueError) when a sentence id is missing or found twice, or when a
capitalized bracket that a template uses is not in the bracket table.

fill(template, values, specials) writes a template's brackets: a table bracket or a one-word bracket ([S], [s],
[check], ...) from `values`, and a longer placeholder ([one clause per missed mark], ...) from `specials`, replaced as
a whole. Text put in is never read again for brackets. A bracket of the template that is not filled is a ValueError,
so a sentence never leaves the code with a '[' of its template behind. The literal "TODO(DECISION.md: ...)" marks of
the draft are not brackets: they stay in the sentence until DECISION.md fills them (the freeze refuses while any
remains).

fmt(x, kind) writes a number as section 14.5 says: one decimal, a minus sign when negative and no plus sign, "-0.0"
written "0.0", "not available" for a number that cannot be computed. Standard library only.
"""
import sys

sys.dont_write_bytecode = True

import math  # noqa: E402
import re  # noqa: E402

NOT_AVAILABLE = "not available"
SOURCE = "PREREGISTRATION.md section 14"

# Every sentence id, in the order of the document, with its anchor: (id, subsection, kind, anchor). kinds: 'quote'
# (the blockquote after a bold label), 'item' / 'string' (the n-th quoted list item or string after a lead-in line),
# 'bullet' (a bullet of section 14.4 that starts with a bold label).
_ANCHORS = (
    ("c1.pass", "14.1", "quote", "All three marks pass:"),
    ("c1.fail", "14.1", "quote", "Any mark fails:"),
    ("c1.clause.net", "14.1", "item", ("The clauses:", 0)),
    ("c1.clause.keeps", "14.1", "item", ("The clauses:", 1)),
    ("c1.clause.even_pull", "14.1", "item", ("The clauses:", 2)),
    ("c1.clause.failed_seed", "14.1", "item", ("The clauses:", 3)),
    ("c1.clause.refused_seed", "14.1", "item", ("The clauses:", 4)),
    ("c1.invalid", "14.1", "quote", "Validity fails:"),
    ("c1.reason.harm", "14.1", "string", ("The reasons:", 0)),
    ("c1.reason.past80", "14.1", "string", ("The reasons:", 1)),
    ("c2.holds", "14.2", "quote", "Holds (every twin beaten):"),
    ("c2.fails", "14.2", "quote", "Fails:"),
    ("c2.clause.twin", "14.2", "item", ("The clauses:", 0)),
    ("c2.clause.refused_seed", "14.2", "item", ("The clauses:", 1)),
    ("c2.each.holds", "14.2", "quote", "Each separately, a twin beaten:"),
    ("c2.each.fails", "14.2", "quote", "Each separately, a twin not beaten:"),
    ("c2.dropped", "14.2", "quote", "Dropped by the noise floor:"),
    ("c2.not_fixed", "14.2", "quote", "Margins not fixed:"),
    ("c2.failed_seed", "14.2", "quote", "If a seed failed:"),
    ("c2.every_seed_failed", "14.2", "quote", "If every seed failed:"),
    ("count.reported", "14.3", "quote", "Always, for the pooled groups:"),
    ("stop.reproduction", "14.4", "bullet", "Reproduction fails:"),
    ("stop.refused", "14.4", "bullet", "BCA refuses on a seed-world:"),
    ("stop.phase", "14.4", "bullet", "A phase stops:"),
)
SENTENCE_IDS = tuple(a[0] for a in _ANCHORS)
FILL_RULES_SECTION = "14.5"

CAPITAL_BRACKET = re.compile(r"\[([A-Z][A-Za-z0-9]*)\]")
SIMPLE_BRACKET = r"\[([A-Za-z0-9]+)\](%?)"
TODO_MARK = re.compile(r"TODO\(DECISION\.md[^)]*\)")


# --------------------------------------------------------------------------------------------- number writing

def fmt(x, kind="points"):
    """A number as section 14.5 writes it. kinds: 'points' ("{:.1f}"), 'percent' (x * 100, "{:.1f}"), 'count' (a
    whole number), 'value' ("{:.4g}", four significant digits). None, NaN and infinities are "not available"; "-0.0"
    is written "0.0" (and "-0" "0"). A string is returned as it is."""
    if isinstance(x, str):
        return x
    if x is None:
        return NOT_AVAILABLE
    if isinstance(x, bool):
        return "yes" if x else "no"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return NOT_AVAILABLE
    if not math.isfinite(v):
        return NOT_AVAILABLE
    if kind == "count":
        return str(int(round(v)))
    if kind == "percent":
        v = v * 100.0
        kind = "points"
    if kind == "points":
        s = f"{v:.1f}"
        return "0.0" if s == "-0.0" else s
    if kind == "value":
        s = f"{v:.4g}"
        return "0" if s == "-0" else s
    raise ValueError(f"unknown number kind {kind!r}")


def join_items(items):
    """"A", "A and B", "A, B and C" (section 14.5)."""
    items = [str(i) for i in items]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


# --------------------------------------------------------------------------------------------- filling

def fill(template, values, specials=None):
    """The template with its brackets written (module docstring). values: {bracket name: str or number} (a number is
    written with fmt(x), points); specials: {placeholder text, brackets included: str}. "[X]%" with X not available is
    written "not available" (no percent sign). Raises ValueError when a bracket of the template is left unfilled."""
    specials = dict(specials or {})
    parts = [re.escape(k) for k in sorted(specials, key=len, reverse=True)]
    rx = re.compile("|".join(parts + [SIMPLE_BRACKET]))
    out, pos, missing = [], 0, []
    for m in rx.finditer(template):
        out.append(template[pos:m.start()])
        token = m.group(0)
        if token in specials:
            out.append(str(specials[token]))
        else:
            name, pct = m.group(1), m.group(2)
            if name not in values:
                missing.append(token)
                out.append(token)
            else:
                s = fmt(values[name])
                out.append(s if (not pct or s == NOT_AVAILABLE) else s + pct)
        pos = m.end()
    out.append(template[pos:])
    literal = rx.sub("", template)
    if missing or "[" in literal or "]" in literal:
        left = missing + re.findall(r"\[[^\]]*\]?", literal)
        raise ValueError(f"unfilled brackets {left} in: {template[:90]}...")
    return "".join(out)


def todo_marks(text):
    """The literal DECISION.md marks left in a text (the freeze refuses while any remains in the documents)."""
    return TODO_MARK.findall(str(text or ""))


# --------------------------------------------------------------------------------------------- extraction

def _section(lines, start_prefix, stop_prefix):
    try:
        i = next(k for k, line in enumerate(lines) if line.startswith(start_prefix))
    except StopIteration:
        raise ValueError(f"no line starts with {start_prefix!r}") from None
    j = next((k for k in range(i + 1, len(lines)) if lines[k].startswith(stop_prefix)), len(lines))
    return lines[i:j]


def _paragraph(lines, i):
    """The paragraph that starts at line i (up to a blank line), its lines stripped and joined by one space."""
    out = []
    while i < len(lines) and lines[i].strip():
        out.append(lines[i].strip())
        i += 1
    return " ".join(out), i


def _unquote(text):
    t = text.strip()
    if len(t) < 2 or not (t.startswith('"') and t.endswith('"')):
        raise ValueError(f"not a quoted sentence: {t[:60]!r}")
    return t[1:-1]


def _outer_quoted(text):
    """The text between the first and the last double quote."""
    a, b = text.find('"'), text.rfind('"')
    if a < 0 or b <= a:
        raise ValueError(f"no quoted text in: {text[:60]!r}")
    return text[a + 1:b]


def _list_items(lines, i):
    """The '- ' list items that follow line i (after optional blank lines), each with its continuation lines joined
    by one space."""
    k = i + 1
    while k < len(lines) and not lines[k].strip():
        k += 1
    items = []
    while k < len(lines) and lines[k].startswith("- "):
        cur = [lines[k][2:].strip()]
        k += 1
        while k < len(lines) and lines[k].strip() and not lines[k].startswith("- ") and lines[k].startswith(" "):
            cur.append(lines[k].strip())
            k += 1
        items.append(" ".join(cur))
    return items


def _blockquote_after(lines, i):
    """The blockquote that follows the bold label on line i: lines starting with '>' (after blank lines), markers
    removed, joined by one space, outer quotes removed."""
    k = i + 1
    while k < len(lines) and not lines[k].strip():
        k += 1
    quoted = []
    while k < len(lines) and lines[k].startswith(">"):
        quoted.append(lines[k][1:].strip())
        k += 1
    if not quoted:
        raise ValueError(f"no blockquote after {lines[i][:60]!r}")
    return _unquote(" ".join(quoted))


def _find_lines(sub, predicate, what):
    hits = [k for k, line in enumerate(sub) if predicate(line)]
    if len(hits) != 1:
        raise ValueError(f"{what}: found {len(hits)} times (exactly once needed)")
    return hits[0]


def _bracket_names(cell):
    return [x.strip() for x in re.split(r",| and ", cell) if x.strip()]


def extract_section14(md_text):
    """Section 14 of the pre-registration as a block (module docstring). Refuses (ValueError) a missing or doubled
    sentence, and a capitalized bracket used in a template that the bracket table does not define."""
    lines = md_text.split("\n")
    sec = _section(lines, "## 14.", "## 15.")
    head_end = next((k for k, line in enumerate(sec) if line.startswith("### 14.")), len(sec))
    head = sec[:head_end]

    i = _find_lines(head, lambda s: s.startswith("The code fills in the brackets."), "the intro paragraph")
    intro, _ = _paragraph(head, i)

    brackets = []
    for line in head:
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 2 or set(cells[0]) <= set("-: ") or cells[0] == "bracket":
            continue
        brackets.append(dict(names=_bracket_names(cells[0]), meaning=cells[1]))
    if not brackets:
        raise ValueError("section 14 has no bracket table")

    i = _find_lines(head, lambda s: s.startswith("**Always attached:**"), "the always-attached sentence")
    text, _ = _paragraph(head, i)
    always = _outer_quoted(text[len("**Always attached:**"):])
    i = _find_lines(head, lambda s: s.startswith("**On the plumbing's seed 0,**"), "the smoke prefix")
    text, _ = _paragraph(head, i)
    prefix = _outer_quoted(text)

    subs = {}
    for k in range(head_end, len(sec)):
        line = sec[k]
        if line.startswith("### 14."):
            num = line.split()[1]
            j = next((m for m in range(k + 1, len(sec)) if sec[m].startswith("### 14.")), len(sec))
            subs[num] = (line[4:].strip(), sec[k:j])

    sentences = {}
    for sid, num, kind, anchor in _ANCHORS:
        if num not in subs:
            raise ValueError(f"section {num} is missing (needed for {sid})")
        _, sub = subs[num]
        if kind == "quote":
            label = f"**{anchor}**"
            i = _find_lines(sub, lambda s, lb=label: s.strip() == lb, f"{sid} ({label} in {num})")
            text = _blockquote_after(sub, i)
            lab = anchor
        elif kind in ("item", "string"):
            lead, n = anchor
            i = _find_lines(sub, lambda s, ld=lead: s.startswith(ld), f"{sid} ({lead!r} in {num})")
            if kind == "item":
                items = _list_items(sub, i)
                if len(items) <= n:
                    raise ValueError(f"{sid}: item {n + 1} after {lead!r} is missing")
                text = _outer_quoted(items[n])
                pre = items[n][:items[n].find('"')].strip()
                lab = f"{lead} item {n + 1}" + (f" ({pre.rstrip(':')})" if pre else "")
            else:
                para, _ = _paragraph(sub, i)
                found = re.findall(r'"([^"]*)"', para)
                if len(found) <= n:
                    raise ValueError(f"{sid}: string {n + 1} after {lead!r} is missing")
                text = found[n]
                lab = f"{lead} string {n + 1}"
        elif kind == "bullet":
            label = f"- **{anchor}"
            i = _find_lines(sub, lambda s, lb=label: s.startswith(lb), f"{sid} ({anchor!r} in {num})")
            item = _list_items(sub, i - 1)[0] if i > 0 else None
            if item is None:
                raise ValueError(f"{sid}: no bullet")
            bold_end = item.find(":**")
            if bold_end < 0:
                raise ValueError(f"{sid}: the bullet's bold label has no ':**'")
            lab = item[2:bold_end + 1]
            text = _outer_quoted(item[bold_end + 3:])
        else:  # pragma: no cover - the table above has no other kind
            raise ValueError(kind)
        if sid in sentences:
            raise ValueError(f"{sid} found twice")
        sentences[sid] = dict(section=num, label=lab, text=text)
    missing = [s for s in SENTENCE_IDS if s not in sentences]
    if missing:
        raise ValueError(f"sentence ids missing: {missing}")

    if FILL_RULES_SECTION not in subs:
        raise ValueError(f"section {FILL_RULES_SECTION} (how the code fills in the sentences) is missing")
    _, sub = subs[FILL_RULES_SECTION]
    first = next((k for k, line in enumerate(sub) if line.startswith("- ")), None)
    if first is None:
        raise ValueError(f"section {FILL_RULES_SECTION} has no bullets")
    fill_rules = _list_items(sub, first - 1)

    defined = {n for row in brackets for n in row["names"]}
    used = set()
    for s in list(sentences.values()) + [dict(text=always)]:
        used.update(CAPITAL_BRACKET.findall(s["text"]))
    if not used <= defined:
        raise ValueError(f"brackets used but not defined in the bracket table: {sorted(used - defined)}")

    return {"source": SOURCE, "intro": intro, "brackets": brackets, "always_attached": always,
            "smoke_prefix": prefix, "sentences": sentences, "fill_rules": fill_rules,
            "todo_marks": sorted({m for s in sentences.values() for m in todo_marks(s["text"])}
                                 | set(todo_marks(always)))}


def check_templates(block):
    """Refuses (ValueError) a claims block that lacks a sentence id or a part the looks read."""
    for key in ("intro", "brackets", "always_attached", "smoke_prefix", "sentences", "fill_rules"):
        if key not in (block or {}):
            raise ValueError(f"the claims block has no {key!r}")
    missing = [s for s in SENTENCE_IDS if s not in block["sentences"]]
    if missing:
        raise ValueError(f"the claims block lacks sentences {missing}")
    return block


def table_bracket_names(block):
    return [n for row in block["brackets"] for n in row["names"]]


if __name__ == "__main__":
    import json
    from pathlib import Path
    doc = Path(__file__).resolve().parent.parent / "PREREGISTRATION.md"
    print(json.dumps(extract_section14(doc.read_text(encoding="utf8")), indent=1, sort_keys=True))

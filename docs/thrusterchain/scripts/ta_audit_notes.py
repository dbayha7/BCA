"""Manual judgements for ta_audit.py (the rerun after the tooltip text pass), made after reading every flagged text.

MANUAL: keys "id:<record id>" (several joined with commas) or a substring of the record's title or text; values are
(type, detail) pairs, where type "drop" removes the automatic findings whose detail contains the given text (a false
hit of the heuristic checks), and any other type adds a finding. After the pass no manual drop is needed: the false
hits of the first rerun were fixed in the checker (five verbs it lacked, "so far" read as a clause) or in the text.
KEPT(rec): what a record keeps on purpose, and why; those findings move from "problems" to "kept".
"""
import re

MANUAL = {}

CODE_TABLES = re.compile(r"^row-T(44|45|46)-")          # Arguments, Registered ids, Names used in the code
CODE_ROWS = {"row-T09-r02", "row-T09-r05"}             # the environment summary's "import" and "Code" rows

WHY_CODE = ("this table's purpose is to give the names the program itself uses (the Arguments and Registered ids "
            "tables, the Names used in the code table, and the import and Code rows of the environment summary), so "
            "the tuple shows them exactly as the code spells them; the column description says they are code, and "
            "the note under the tuple explains the types, files, folders and capital-letter names they contain")
WHY_HASH = ("the table itself shortens these hashes to their first 8 characters followed by …; the tuple shows the "
            "cell exactly as the table does and never cuts it further, and its note says that only the first "
            "characters of the hash are shown")
WHY_TITLE = ("this is the page's own <title> (ThrusterChain), the first line of the fragment; the test harness puts the "
             "fragment in a body, so the harvest finds it there, but publishing places it in the document head, where "
             "it names the browser tab and shows no tooltip")


def KEPT(rec):
    rid = rec["id"]
    if CODE_TABLES.match(rid) or rid in CODE_ROWS:
        return {"code", "acronym"}, WHY_CODE
    if rid in ("row-T41-r04", "row-T41-r07"):
        return {"truncation"}, WHY_HASH
    if rid == "native" and rec["text"] == "ThrusterChain":
        return {"unclear"}, WHY_TITLE
    return set(), ""

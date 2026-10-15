"""Read-only: which tooltip-source texts (ta_code.py snippets, ta_texts.py records) still appear on a page.
usage: python3 scripts/p4_e3_srccheck.py PAGE"""
import html, json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ta_code as C, ta_texts as T
page = open(sys.argv[1], encoding="utf-8").read()
plain = html.unescape(page)
for lst in ("TT", "DR", "IW"):
    for item in getattr(C, lst):
        name, new = item[0], item[2]
        if new not in page:
            print(f"ta_code.{lst}: new snippet not on the page: {name}")
print("CSS_ADD on page:", C.CSS_ADD.strip()[:60] in page)
m = re.search(r"const COLS = (\{.*?\});\n", page, flags=re.S); cols = json.loads(m.group(1))
for sec in T.PASS3_FINAL:
    print("PASS3_FINAL", sec, "same as page:", T.PASS3_FINAL[sec] == cols.get(sec))

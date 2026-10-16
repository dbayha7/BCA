"""Compare a re-run (run_nb.py's JSON) with the committed notebook's stored outputs, line by line per code cell.
Printed lines are the streams plus each display's Markdown (or, without one, its plain text; images are not counted).
Prints the counts of identical and differing printed lines and every differing pair (only running times are expected
to differ). Usage: python compare_nb.py STORED.ipynb RERUN.json"""
import json
import sys

nb = json.load(open(sys.argv[1], encoding="utf-8"))
run = {c["cell"]: c for c in json.load(open(sys.argv[2], encoding="utf-8"))["cells"]}
same = diff = missing = 0
pairs = []
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    st = []
    for o in c.get("outputs", []) or []:
        if o.get("output_type") == "stream":
            st.append("".join(o.get("text", [])))
        else:   # a display: its Markdown if it has one, else its plain text (images are not counted)
            dd = o.get("data", {})
            if "text/markdown" in dd:
                st.append("".join(dd["text/markdown"]) + "\n")
            elif "text/plain" in dd and "image/png" not in dd:
                st.append("".join(dd["text/plain"]) + "\n")
    rr = []
    if i not in run:
        missing += 1
        continue
    if run[i]["status"] != "ok":
        print("cell", i, "status", run[i]["status"])
    for o in run[i]["outputs"]:
        if o["type"] == "stream":
            rr.append(o["text"])
        elif o.get("markdown") is not None:
            rr.append(o["markdown"] + "\n")
        elif o.get("plain") is not None:
            rr.append(o["plain"] + "\n")
    a, b = "".join(st).splitlines(), "".join(rr).splitlines()
    for k in range(max(len(a), len(b))):
        x = a[k] if k < len(a) else None
        y = b[k] if k < len(b) else None
        if x == y:
            same += 1
        else:
            diff += 1
            pairs.append((i, x, y))
print(f"{sys.argv[1].split('/')[-1]}: {same} identical printed lines, {diff} differ, {missing} code cells not run")
for i, x, y in pairs:
    print(f"  cell {i}: stored {x!r}\n          rerun  {y!r}")

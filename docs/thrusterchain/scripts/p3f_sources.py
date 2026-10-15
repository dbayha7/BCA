"""Pass 3, final fixes: bring the tooltip text sources in line with the page (run once, after p3f_apply.py --write).

  wsl.exe -d Ubuntu-22.04 -u root -e bash -lc 'python3 /mnt/c/Users/David/dev/bca-shift-work/page/scripts/p3f_sources.py'

Literal, counted replacements in ta_texts.py and ta_code.py for the texts this pass changed where the sources hold them
as plain strings; then ta_texts.py gets a record, PASS3_FINAL, of the column descriptions of the four sections whose
tables changed, copied from the page (ta_apply.py does not use it).
"""
import json
import os
import pprint

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = "/mnt/c/Users/David/dev/bca-shift-work/page/thrusterchain.html"

EDITS = {
    "ta_texts.py": [
        ("critic rates highest, on the same scale as the caution multiplier m. It is μ = dλκ / 2, where ",
         "critic rates highest, on the same scale as the dose m (the caution multiplier on the pull toward the "
         "recorded command). It is μ = dλκ / 2, where ", 1),
        ('"The test lean is λ = −2, the same as in training', '"The test lean is ξ = −2, the same as in training', 1),
        ('"With the lean λ = 0, every current', '"With the lean ξ = 0, every current', 1),
        ('"With the lean λ = +2, strong currents', '"With the lean ξ = +2, strong currents', 1),
        ('"The lean λ sets how far the test\'s distribution of currents leans toward strong currents. "\n'
         '                      "Training uses λ = −2."',
         '"The lean ξ sets how far the test\'s distribution of currents leans toward strong currents. "\n'
         '                      "Training uses ξ = −2."', 1),
        ('("λ (lambda)", "The lean λ (lambda) says how far the current\'s distribution leans toward strong currents. Its "\n'
         '     "density is proportional to e to the power λc, over 0.1 to 1.1. Training uses λ = −2."),',
         '("ξ (xi)", "The lean ξ (xi) says how far the current\'s distribution leans toward strong currents. Its "\n'
         '     "density is proportional to e to the power ξc, over 0.1 to 1.1. Training uses ξ = −2."),', 1),
        ("the speed number of this float gets a single kick of 1,", "the speed number of this float gets a single push of 1,",
         1),
    ],
    "ta_code.py": [
        ("`Test lean λ: ${sgn(l, 2)}", "`Test lean ξ: ${sgn(l, 2)}", 1),
    ],
}


def main():
    for name, edits in EDITS.items():
        path = os.path.join(HERE, name)
        s = open(path, encoding="utf-8").read()
        for old, new, n in edits:
            assert s.count(old) == n, f"{name}: {s.count(old)} x {old[:70]!r}"
            s = s.replace(old, new)
        open(path, "w", encoding="utf-8", newline="").write(s)
        print(name, len(edits), "edits")
    page = open(PAGE, encoding="utf-8").read()
    head = "  const COLS = "
    i = page.index(head)
    cols = json.loads(page[i + len(head):page.index(";\n", i)].replace("<\\/", "</"))
    rec = {sid: cols[sid] for sid in ("results", "weighting-test", "dynamics", "methods")}
    path = os.path.join(HERE, "ta_texts.py")
    s = open(path, encoding="utf-8").read()
    assert "PASS3_FINAL" not in s, "already recorded"
    block = ("\n\n# ------------------------------------------------------------------------------------- pass 3, final fixes\n"
             "# The column descriptions of Results, the weighting test, Transition Dynamics and Methods as the page now has\n"
             "# them (p3f_apply.py wrote them; ta_apply.py does not use this record). Also changed on the page: the μ tooltip\n"
             "# (the dose m), the lean λ -> ξ in the weighting test, and the glossary rules for the data-only penalties\n"
             "# and the forecast's fixed imitations.\n"
             "PASS3_FINAL = " + pprint.pformat(rec, width=118, sort_dicts=False) + "\n")
    open(path, "w", encoding="utf-8", newline="").write(s.rstrip("\n") + block)
    ns = {}
    exec(compile(open(path, encoding="utf-8").read(), path, "exec"), ns)
    assert ns["PASS3_FINAL"] == rec
    print("ta_texts.py: PASS3_FINAL recorded for", ", ".join(rec))


if __name__ == "__main__":
    main()

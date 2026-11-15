"""Judge marks K1 to K3 for standard BCA (PROTOCOL.md, Amendment 11) from the saved policies. NumPy only.

    python -I -B judge_standard_bca.py --label standard_bca_published

This is judge_version_a.py with standard BCA (the learner's `bca` arm) in Version A's place and
`shuffled_width` as its shuffled control. The marks, the scoring and the tables are that file's,
unchanged. The outputs are then renamed to standard_bca.* and their labels corrected.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import judge_version_a as J  # noqa: E402

STANDARD_BCA, CONTROL = "bca", "shuffled_width"
RELABEL = (("version_a_helps_on_the_published_map", "standard_bca_helps_on_the_published_map"),
           ("judge_version_a.py", "judge_standard_bca.py"), ("Version A", "Standard BCA"))

if __name__ == "__main__":
    J.VERSION_A, J.CONTROL = STANDARD_BCA, CONTROL
    J.NAMES = {J.BASELINE: "Baseline", STANDARD_BCA: "Standard BCA", CONTROL: "Shuffled control",
               J.DECISIVE: "Decisive baseline", J.SHARPENED: "Scaled baseline (no mark)"}
    J.main()
    label = sys.argv[sys.argv.index("--label") + 1]
    folder = HERE / "results" / label
    for old, new in (("version_a.json", "standard_bca.json"), ("version_a.md", "standard_bca.md"),
                     ("version_a_records.json", "standard_bca_records.json")):
        text = (folder / old).read_text(encoding="utf-8")
        for before, after in RELABEL:
            text = text.replace(before, after)
        (folder / new).write_text(text, encoding="utf-8")
        (folder / old).unlink()
    print(f"written to results/{label}/standard_bca.md, standard_bca.json, standard_bca_records.json")
